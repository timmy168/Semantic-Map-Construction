import numpy as np
from PIL import Image
import numpy as np
import json
import os
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import argparse

cam_extr = []

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8) # 0~255
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def lable_semantic(img):
    with open("replica_v1/apartment_0/habitat/info_semantic.json", 'r') as f:
    # with open(os.path.join(os.path.normpath("apartment_0"), 'habitat', 'info_semantic.json'), 'r') as f:
        scene_dict = json.load(f)
    instance_id_to_semantic_label_id = np.array(scene_dict["id_to_label"])
    semantic = instance_id_to_semantic_label_id[img]
    return semantic

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    
    # a BEV_RGB visual sensor, to the agent
    bev_rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    bev_rgb_sensor_spec.uuid = "bev_color_sensor"
    bev_rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    bev_rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    bev_rgb_sensor_spec.position = [0.0, settings["sensor_height"], -1.5]
    bev_rgb_sensor_spec.orientation = [
        -np.pi/2,
        0.0,
        0.0,
    ]
    bev_rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    
    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec,bev_rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)
        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("#############################")
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  
              sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

        cam_extr.append([sensor_state.position[0], sensor_state.position[1], sensor_state.position[2], 
                    sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z])

        #return front_rgb,front_depth,front_semantic_label 
        return transform_rgb_bgr(observations["color_sensor"]), transform_depth(observations["depth_sensor"]), lable_semantic(observations["semantic_sensor"])

#save front_rgb,depth_rgb,front_semantic_label while moving onece
def save_reconstruct(save_img, count):

    # create and check the folder
    output_folder = 'reconsrtuct_data'
    image_folder = os.path.join(output_folder, 'images')
    depth_folder = os.path.join(output_folder, 'depth')
    annotations_folder = os.path.join(output_folder, 'annotations')

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

    # save the image
    cv2.imwrite(os.path.join(image_folder, str(count) + '.png'), save_img[0])
    cv2.imwrite(os.path.join(depth_folder, str(count) + '.png'), save_img[1])
    cv2.imwrite(os.path.join(annotations_folder, str(count) + '.png'), save_img[2])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)  
    args = parser.parse_args()

    # Scene path
    test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"
    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height":1.5 ,  # Height of sensors in meters, relative to the agent1.5 /1:27
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "sensor_pitch": 0,  # sensor pitch (x rotation in rads)(- is down,+is up)
    }

    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Set agent state
    agent_state = habitat_sim.AgentState()

    if args.floor == 1:
        agent_state.position = np.array([0.0, 0.0, 0.0]) # agent in world space
    elif args.floor == 2:
        agent_state.position = np.array([0.0, 1.0, -1.0]) # agent in world space

    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)

    FORWARD_KEY="w"
    LEFT_KEY="a"
    RIGHT_KEY="d"
    SAVE_BEV_IMG="z"
    FINISH="f"
    print("----------------------------------")
    print("use keyboard to control the agent")
    print(" w for go forward  ")
    print(" a for turn left  ")
    print(" d for trun right  ")
    print(" f for finish and quit the program")
    print("----------------------------------")

    count = 1 
    action = "move_forward"
    save_img=navigateAndSee(action)                
    save_reconstruct(save_img,count)                
    count=count+1                                  

    while True:
        keystroke = cv2.waitKey(0)                  
        if keystroke == ord(FORWARD_KEY):          
            action = "move_forward"
            save_img=navigateAndSee(action)         
            save_reconstruct(save_img,count)        
            count=count+1                          
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
            save_img=navigateAndSee(action)
            save_reconstruct(save_img,count) 
            count=count+1  
            print("action: LEFT")     
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
            save_img=navigateAndSee(action)
            save_reconstruct(save_img,count) 
            count=count+1  
            print("action: RIGHT")  
        elif keystroke == ord(FINISH):
            count=count-1
            print("action: FINISH ")
            print("All image number: ",count)                            
            break
        else:
            print("INVALID KEY")
            continue
    
    np.save('reconsrtuct_data/GT_pose.npy', np.asarray(cam_extr))