import time
import mujoco as mj
import mujoco.viewer as mjViewer

paused = False


def key_callback(keycode: int):
    if chr(keycode) == " ":
        global paused
        paused = not paused


model = mj.MjModel.from_xml_path("models/ant.xml")
data = mj.MjData(model)

with mjViewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    start = time.time()
    while viewer.is_running():
        if paused:
            continue

        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mj.mj_step(model, data)

        # Example modification of a viewer option: toggle contact points every two seconds.
        # with viewer.lock():
        #     viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
