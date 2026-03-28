# RBY1 Robot Teleoperation (Meta Quest + Unity)

This guide shows you how to control the **RBY1 robot** using **Meta Quest** and **Unity** with real-time teleoperation. The Meta Quest sends pose and button data to a PC that translates it into robot commands via the RBY1 SDK.

---

## Requirements

- Meta Quest 2 or 3 (Developer Mode enabled)
- Unity Editor (tested on **Unity 6000.1.0f1**)
- Android Build Support (installed via Unity Hub)
- RBY1 Robot:
  - [RPC >= v0.7.0](https://github.com/RainbowRobotics/rby1-release/releases/tag/v0.7.0)
  - [SDK >= v0.7.0](https://github.com/RainbowRobotics/rby1-sdk/releases/tag/v0.7.0)

---

## Step 1: Enable Developer Mode on Meta Quest

Follow the official Meta guide here: [Meta Quest - Enable Developer Mode](https://developers.meta.com/horizon/documentation/native/android/mobile-device-setup/)

1. Log in to the **[Meta Developer site](https://developer.oculus.com/manage/organizations/)** and create an organization if needed.
2. On your smartphone:
   - Install the **Meta Quest mobile app**
   - Go to **Devices > Developer Mode** and enable it
3. Connect the headset to your PC via USB and accept USB debugging permission.

---

## Step 2: Unity Setup for Meta Quest

> Unity must be installed with **Android Build Support**. This setup assumes a clean project using the **Universal Render Pipeline (URP)** or **3D (Core)** template.

### 2.1 Create Unity Project

- Open Unity Hub → `New Project`
- Template: **Universal 3D**
- Project name: e.g. `RBY1_Teleoperation`

### 2.2 Import Assets

- Download and import: [MetaQuestPoseReader.unitypackage](./unity/MetaQuestPoseReader.unitypackage)
- Drag it into Unity or use `Assets > Import Package > Custom Package`

### 2.3 Add Required Packages

Go to `Window > Package Manager` and install the following:

- **XR Plug-in Management** (`com.unity.xr.management`)
- **Oculus XR Plugin** (`com.unity.xr.oculus`)
- **XR Interaction Toolkit** (`com.unity.xr.interaction.toolkit`)
- **Newtonsoft Json** (`com.unity.nuget.newtonsoft-json`)

> Tip: Use "Add package by name..." if you can't find them via search.
> - com.unity.xr.management
> - com.unity.xr.oculus
> - com.unity.xr.interaction.toolkit
> - com.unity.nuget.newtonsoft-json

Click `Window > TextMeshPro > Import TMP Essential Resources` to install **TextMesh Pro**

### 2.4 Configure XR Settings

- `Edit > Project Settings > XR Plug-in Management` → enable **Oculus** under **Android**

### 2.5 Scene Setup

- Drag `Scenes/MainScene` into **Hierarchy**
- Remove `SampleScene`

### 2.6 Build Settings

- `File > Build Profiles`
  - `Scene List`
    - Add open scenes 
    - Disable `Scenes/SampleScenes`
  - Platform: `Android` → **Switch Platform**
  - Minimum API Level: `Android 10.0 (API 29)`

### 2.7 Build and Run

- Connect Meta Quest via USB
- Click **Build and Run**
- Save as `.apk` and deploy

## Step 3: Run the Python Teleoperation Script

> Scripts are located in the [`python/`](./python) folder.

### 3.1 Python Dependencies

Install the required packages:

```bash
pip install rby1-sdk==0.7.0 pyzmq>=26.4.0,<27.0.0 scipy>=1.15.3,<2.0.0
```

### 3.2 Example Run Command

```bash
python scripts/teleop.py \
  --local_ip 172.16.120.136 \
  --meta_quest_ip 172.16.120.178 \
  --rby1 192.168.30.1:50051 \
  --rby1_model "m" \
  --no_head

python scripts/eval.py
python scripts/eval_molmoact.py

python scripts/collect.py \
  --local_ip 172.16.120.136 \
  --meta_quest_ip 172.16.120.178 \
  --rby1 192.168.30.1:50051 \
  --rby1_model "m" \
  --no_head

python scripts/replay.py \
  --task task_dir_name \
  --episode episode_num \ 
  --mode robot \
  --no_head
```

#### Option Descriptions

| Option            | Description                                                                        |
| ----------------- | ---------------------------------------------------------------------------------- |
| `--local_ip`      | IP address of the PC running the Python script                                     |
| `--meta_quest_ip` | IP address shown inside Meta Quest app (usually shown on screen)                   |
| `--rby1`          | IP and port of the RBY1 RPC server (format: `IP:PORT`, e.g., `192.168.30.1:50051`) |
| `--rby1_model`    | RBY1 model  (e.g., "a")                                                            |
| `--no_gripper`    | (Optional) Disables gripper control. Omit if using a robot with a gripper.         |

> If you are using a UPC with gripper support, omit `--no_gripper`.  
> Omit `--no_gripper` if using a robot with a gripper. (Must be run on UserPC)

---

## Controller Mapping

| Input              | Description                        |
| ------------------ | ---------------------------------- |
| Right A Button     | Reset pose and begin teleoperation |
| Right B Button     | Stop teleoperation                 |
| Right Trigger      | Control right arm                  |
| Left Trigger       | Control left arm                   |
| Both Triggers Held | Control torso                      |

---

## Notes

- Ensure both the Meta Quest and the PC are on the **same Wi-Fi network**.
- The Meta Quest app shows its IP address on screen when running.

---

# Code Exaplanation

## Cartesian Impedance Control: Key Concepts

This control mode is used to guide the **torso** and **arms** of the RBY1 robot in Cartesian space with compliant motion.

---

### Command Builder Overview

```python
builder = (
    rby.CartesianImpedanceControlCommandBuilder()
    .set_joint_stiffness([...])
    .set_joint_torque_limit([...])
    .add_joint_limit("joint", min, max)
    .add_target(reference_link_name, 
                target_link_name, 
                target_pose, 
                linear_velocity, 
                angular_velocity, 
                linear_acceleration, 
                angular_acceleration)
)
```

- **Stiffness**: Joint impedance gain
- **Torque Limit**: Prevents the robot from exerting excessive force
- **Joint Limit**: Motion boundaries per joint
- **Target**: Pose to reach in Cartesian space 

---

### Example: Right Arm

```python
right_builder = (
    rby.CartesianImpedanceControlCommandBuilder()
    .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])         # 7 Joints (Joint 0–5: 80 Nm/rad, Joint 6: 40 Nm/rad)
    .set_joint_torque_limit([30]*7)             # Torque clamped to ±30 Nm
    .add_joint_limit("right_arm_3", -2.6, -0.5) # Prevent overstretch / singularity
    .add_target("base", "link_right_arm_6", right_T, 2, np.pi*2, 20, np.pi*80)
          # move "link_right_arm_6" link to `right_T` with respect to "base"
          # (linear velocity limit: 2 m/s,
          #  angular velocity limit: 2π rad/s,
          #  linear acceleration limit: 20 m/s²,
          #  angular acceleration limit: 80π rad/s²)
)
```

---

### Composition

```python
ctrl_builder = (
    rby.BodyComponentBasedCommandBuilder()
    .set_torso_command(torso_builder)
    .set_right_arm_command(right_builder)
    .set_left_arm_command(left_builder)
)
```

> Tip: When `--whole_body` is used, the torso, right arm, and left arm targets are added into a **single CartesianImpedanceControlCommandBuilder**, instead of splitting by body part.

This `ctrl_builder` is passed into the robot control stream:

```python
stream.send_command(
    rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder()
        .set_body_command(ctrl_builder)
        ...
    )
)
```
