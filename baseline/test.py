from goleftenv import GoLeftEnv
env = GoLeftEnv(grid_size=10)

obs, _ = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

GO_LEFT = 0
# Hardcoded best agent: always go left!
n_steps = 20
for step in range(n_steps):
    print(f"Step {step + 1}")
    obs, reward, terminated, truncated, info = env.step(GO_LEFT)
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break