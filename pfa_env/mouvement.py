import pybullet as p
import pybullet_data
import time

def main():
    JOINTS_ID = [
        "L1_J1",
        "L1_J2",
        "L1_J3",
        "L2_J1",
        "L2_J2",
        "L2_J3",
        "L3_J1",
        "L3_J2",
        "L3_J3",
        "L4_J1",
        "L4_J2",
        "L4_J3",
    ]
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", physicsClientId=client)
    spider = p.loadURDF(
        "urdf/Spider_URDF.urdf",
        [0, 0, 0.1],
        physicsClientId=client,
    )
    while 1:
        try:
            p.stepSimulation(client)
            time.sleep(1./240.)
        except KeyboardInterrupt:
            p.disconnect()

if __name__ == "__main__":
    main()