import os
# Set dummy driver BEFORE pygame is initialized (which happens in MazerushEnv -> MazerushRenderer)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import time
import cv2
import yaml
import numpy as np
import pygame
import huggingface_hub
from flask import Flask, render_template, Response, jsonify, request
from mazerush_env import MazerushEnv
from agent_utils import build_agents, DeepQAgent, HumanAgent

app = Flask(__name__)

# --- Hardcoded Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config/mazerush_large.yaml")
RESUME_TYPE = "hf"  # "hf" or "local"
RESUME_PATHS = os.path.join(BASE_DIR, "out/20260228_100838/player0_ckpt_4000.pt")
HF_MODEL_NAME = "20260228_101239__player0_ckpt_114000"
RENDER_MODE = "human"
# -------------------------------

# Global environment state
env = None
agents = []
config = None

def get_env():
    global env, agents, config
    if env is None:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        
        render_mode = RENDER_MODE
        if RESUME_TYPE == "hf":
            resume_paths = huggingface_hub.hf_hub_download(repo_id=f"marvinvibe/{HF_MODEL_NAME}", filename=HF_MODEL_NAME + ".pt")
        else:
            resume_paths = RESUME_PATHS or config.get("training", {}).get("resume_paths")

        env_config = config["env"].get("config") or {}
        player_configs = config.get("players", [{"type": "HumanAgent"}, {"type": "RandomAgent"}])
        num_players = len(player_configs)

        env = MazerushEnv(num_players=num_players, render_mode=render_mode, **env_config)
        env.reset()
        num_states = env.observation_space.shape[0]
        agent_config = config.get("agent", {})
        agents = build_agents(player_configs, env.action_space, num_states, agent_config)
        for i, agent in enumerate(agents):
            agent.set_player(env.players[i])


        # Resume DeepQAgents
        if resume_paths:
            if isinstance(resume_paths, str):
                resume_paths = resume_paths.split(",")
            resume_idx = 0
            for i, agent in enumerate(agents):
                if isinstance(agent, DeepQAgent):
                    resume_path = resume_paths[resume_idx]
                    resume_idx = (resume_idx + 1) % len(resume_paths)
                    if os.path.exists(resume_path):
                        agent.load(resume_path)
                        print(f"Resumed player {i} from {resume_path}")
                    else:
                        print(f"Warning: Checkpoint not found: {resume_path}")
    return env

def gen_frames():
    global env, agents
    e = get_env()
    # Initialize observations from the current state
    obs_n = [e._get_obs(i) for i in range(e.num_players)]
    
    while True:
        time_start = time.time()
        # Select actions
        action_n = [agent.select_action(obs) for agent, obs in zip(agents, obs_n)]
        
        obs_n, reward, done, truncated, info = e.step(action_n)
        
        if any(done) or any(truncated):
            obs_n, _ = e.reset()

        # Capture the pygame screen
        if e._renderer and e._renderer._screen:
            view = pygame.surfarray.array3d(e._renderer._screen)
            view = view.transpose([1, 0, 2])
            view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            
            ret, buffer = cv2.imencode('.jpg', view)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        if time.time() - time_start < 1.0 / e.fps:
            time.sleep(1.0 / e.fps - (time.time() - time_start))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset', methods=['POST'])
def reset_env():
    global env
    if env:
        env.reset()
    return jsonify({"status": "success"})

@app.route('/key', methods=['POST'])
def handle_key():
    global agents
    e = get_env() # Ensure agents are initialized
    data = request.json
    key = data.get('key')
    
    key_map = {
        'ArrowUp': pygame.K_UP,
        'ArrowDown': pygame.K_DOWN,
        'ArrowLeft': pygame.K_LEFT,
        'ArrowRight': pygame.K_RIGHT,
        ' ': pygame.K_SPACE
    }
    
    if key in key_map:
        for agent in agents:
            if isinstance(agent, HumanAgent):
                event = pygame.event.Event(pygame.KEYDOWN, key=key_map[key])
                agent.key_listener(event)
                return jsonify({"status": "key_received"})
            
    return jsonify({"status": "ignored"})

@app.route('/status')
def get_status():
    global env
    e = get_env() # Ensure env is initialized
    return jsonify({
        "tick": e.tick,
        "num_players": e.num_players,
        "alive": [p.alive for p in e.players]
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
