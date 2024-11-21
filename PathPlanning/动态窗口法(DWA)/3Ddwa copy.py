import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Robot:
    def __init__(self, x, y, z, vx, vy, vz, v_max,v_min, a_max,a_min, dt, alpha=0.3, beta=0.7, gamma=0.1):
        """
        初始化机器人状态和参数

        Args:
            x, y, z: 初始位置
            vx, vy, vz: 初始速度
            v_max: 最大线速度
            a_max: 最大加速度
            dt: 时间步长
            alpha: 目标距离的权重
            beta: 障碍物距离的权重
            gamma: 速度偏好的权重
        """
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.v_max = v_max  # 最大线速度
        self.v_min = v_min  # 最小线速度
        self.a_max = a_max  # 最大线加速度
        self.a_min = a_min  # 最小线加速度
        self.dt = dt  # 时间步长

        # 权重参数
        self.alpha = alpha  # 目标距离权重
        self.beta = beta  # 障碍物距离权重
        self.gamma = gamma  # 速度偏好权重

    def update_state(self, new_position, new_velocity):
        """
        更新机器人的位置和速度

        Args:
            new_position: 新位置 (x, y, z)
            new_velocity: 新速度 (vx, vy, vz)
        """
        self.x, self.y, self.z = new_position
        self.vx, self.vy, self.vz = new_velocity

    def get_state(self):
        """
        获取当前机器人状态

        Returns:
            当前状态: (x, y, z, vx, vy, vz)
        """
        return (self.x, self.y, self.z, self.vx, self.vy, self.vz)

    def set_weights(self, alpha, beta, gamma):
        """
        动态调整权重参数

        Args:
            alpha: 目标距离的权重
            beta: 障碍物距离的权重
            gamma: 速度偏好的权重
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_weights(self):
        """
        获取当前权重参数

        Returns:
            权重参数: (alpha, beta, gamma)
        """
        return (self.alpha, self.beta, self.gamma)



# Generate mixed obstacles
def generate_mixed_obstacles(num_obstacles, space_size, radius_range, height_range):
    obstacles = []
    for _ in range(num_obstacles):
        obstacle_type = np.random.choice(['sphere', 'cylinder'])
        x = np.random.uniform(0, space_size)
        y = np.random.uniform(0, space_size)
        if obstacle_type == 'sphere':
            z = np.random.uniform(0, space_size)  # Spheres can float
            radius = np.random.uniform(radius_range[0], radius_range[1])
            obstacles.append(('sphere', x, y, z, radius, None))
        else:  # Cylinder
            z = 0  # Cylinders start at ground level
            radius = np.random.uniform(radius_range[0], radius_range[1])
            height = np.random.uniform(height_range[0], height_range[1])
            obstacles.append(('cylinder', x, y, z, radius, height))
    return obstacles


# Predict trajectories
def predict_trajectory(robot, vx, vy, vz):
    trajectory = []
    x, y, z = robot.x, robot.y, robot.z
    for _ in range(5):  # Predict for 10 steps
        x += vx * robot.dt
        y += vy * robot.dt
        z += vz * robot.dt
        trajectory.append((x, y, z))
    return np.array(trajectory)

from scipy.interpolate import splprep, splev

def smooth_trajectory(trajectory, smooth_factor=5):
    """
    使用 B 样条平滑轨迹
    Args:
        trajectory: 原始预测轨迹 (N, 3)，每一行是 (x, y, z)
        smooth_factor: 光滑因子，值越大越平滑
    Returns:
        smooth_traj: 光滑后的轨迹
    """
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    
    # 使用 B 样条拟合原始轨迹
    tck, u = splprep([x, y, z], s=smooth_factor)
    
    # 生成更多平滑点
    u_fine = np.linspace(0, 1, len(x) * 10)  # 增加点的密度
    x_smooth, y_smooth, z_smooth = splev(u_fine, tck)
    
    # 返回光滑轨迹
    smooth_traj = np.array([x_smooth, y_smooth, z_smooth]).T
    return smooth_traj

from scipy.interpolate import interp1d

def interpolate_trajectory(trajectory, num_points=100):
    """
    使用线性插值平滑轨迹
    Args:
        trajectory: 原始轨迹 (N, 3)，每一行是 (x, y, z)
        num_points: 插值后点的数量
    Returns:
        smooth_traj: 插值后的平滑轨迹
    """
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    
    # 创建插值函数
    t = np.linspace(0, 1, len(x))
    interp_x = interp1d(t, x, kind='cubic')
    interp_y = interp1d(t, y, kind='cubic')
    interp_z = interp1d(t, z, kind='cubic')
    
    # 插值生成更多点
    t_fine = np.linspace(0, 1, num_points)
    smooth_x = interp_x(t_fine)
    smooth_y = interp_y(t_fine)
    smooth_z = interp_z(t_fine)
    
    smooth_traj = np.array([smooth_x, smooth_y, smooth_z]).T
    return smooth_traj



def evaluate_trajectory(trajectory, goal, obstacles, robot, safe_distance):
    """
    评价给定轨迹的质量，综合考虑目标距离、障碍物安全性和轨迹时间效率。
    
    Args:
        trajectory: UAV的预测轨迹，形状为 (N, 3)。
        goal: 目标点坐标 (x, y, z)。
        obstacles: 障碍物列表，每个障碍物包含 (类型, ox, oy, oz, radius, height)。
        robot: Robot对象，包含评价权重等参数。
        safe_distance: 安全距离，若小于该值则显著降低得分。

    Returns:
        轨迹得分：值越高越优。
    """
    # 权重系数（可以从Robot类中动态调整）
    alpha = robot.alpha  # 目标距离的权重
    beta = robot.beta    # 障碍物距离的权重
    gamma = robot.gamma  # 速度偏好的权重

    # 初始化评价值
    # goal_distance = np.linalg.norm(trajectory[-1] - np.array(goal))  # 目标距离
    min_obstacle_distance = float('inf')  # 障碍物最小距离

    # # 计算轨迹的平均速度
    # total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    # average_speed = total_distance / (len(trajectory) * robot.dt*5)
    # 计算轨迹总距离
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

    # 计算轨迹的总时间
    total_time = (len(trajectory) - 1) * robot.dt

    # 计算平均速度
    average_speed = total_distance / total_time

    # 裁剪平均速度，确保在 [v_min, v_max] 范围内
    average_speed = np.clip(average_speed, robot.v_min, robot.v_max)

    # 归一化速度得分
    speed_score = (average_speed - robot.v_min) / (robot.v_max - robot.v_min)


        # 初始化最小距离
    min_obstacle_distance = float('inf')

    # 遍历轨迹每个点，计算与障碍物的最小距离
    for point in trajectory:
        for obs in obstacles:
            obs_type, ox, oy, oz, radius, height = obs
            if obs_type == 'sphere':
                # 球体障碍物距离计算
                dist = np.linalg.norm(point - np.array([ox, oy, oz])) - radius
            elif obs_type == 'cylinder':
                # 圆柱体障碍物距离计算
                horizontal_dist = np.linalg.norm(point[:2] - np.array([ox, oy])) - radius
                if 0 <= point[2] <= height:
                    dist = horizontal_dist
                else:
                    dist = float('inf')  # 超出高度范围
            else:
                dist = float('inf')  # 未知障碍物类型
            
            # 更新最小距离
            min_obstacle_distance = min(min_obstacle_distance, dist)

    # 根据最小距离计算评分
    if min_obstacle_distance < 0:
        # 在障碍物内部，得分为负数
        obstacle_score = min_obstacle_distance
    elif min_obstacle_distance < safe_distance:
        # 在障碍物外部，但距离小于安全距离
        obstacle_score = min_obstacle_distance / safe_distance
    else:
        # 距离超过安全距离，满分
        obstacle_score = 1.0

        # 计算初始点到目标的最大距离
    max_distance = np.linalg.norm(trajectory[0] - np.array(goal))



    # 起点和终点
    start_point = trajectory[0]  # 起点
    goal_point = np.array(goal)  # 目标点

    # 计算起点到目标点的方向向量
    direction_to_goal = (goal_point - start_point) / np.linalg.norm(goal_point - start_point)

    # 计算在给定时间内可以到达的最远点
    d_max = robot.v_max * robot.dt  # 最大可移动距离
    reachable_point = start_point + direction_to_goal * d_max

    # 计算最远点与目标点的距离
    min_goal_distance = np.linalg.norm(reachable_point - goal_point)

    # 计算轨迹终点与目标点的实际距离
    goal_distance = np.linalg.norm(trajectory[-1] - goal_point)

    # 计算目标得分（归一化）
    goal_score =  min_goal_distance / (goal_distance + 1e-6)  # 防止除零


    # speed_score = average_speed-robot.v_min / (robot.v_max - robot.v_min)  # 归一化速度得分

    # 综合得分（越高越优）
    total_score = alpha * goal_score + beta * obstacle_score + gamma * speed_score
    return total_score
# def evaluate_trajectory(trajectory, goal, obstacles, robot, safe_distance):
#     goal_distance = np.linalg.norm(trajectory[-1] - np.array(goal))
#     min_obstacle_distance = float('inf')

#     for point in trajectory:
#         for obs in obstacles:
#             # Unpack obstacle values
#             obs_type, ox, oy, oz, radius, height = obs
#             if obs_type == 'sphere':
#                 dist = np.linalg.norm(point - np.array([ox, oy, oz])) - radius
#             elif obs_type == 'cylinder':
#                 horizontal_dist = np.linalg.norm(point[:2] - np.array([ox, oy])) - radius
#                 if 0 <= point[2] <= height:
#                     dist = horizontal_dist
#                 else:
#                     dist = float('inf')  # Outside height range
#             else:
#                 dist = float('inf')  # Unknown type

#             # Update the minimum obstacle distance
#             min_obstacle_distance = min(min_obstacle_distance, dist)

#     if min_obstacle_distance < safe_distance:
#         return float('inf')

#     return 1 / (goal_distance + 1e-6)  # Lower goal distance preferred
def predict_smooth_trajectory(robot, vx, vy, vz, ax, ay, az, steps=10):
    """
    使用加速度平滑轨迹预测
    Args:
        robot: 机器人对象，包含当前位置和时间步长
        vx, vy, vz: 初始速度
        ax, ay, az: 加速度
        steps: 预测步数
    Returns:
        trajectory: 平滑的轨迹，形状为 (steps, 3)
    """
    trajectory = []
    x, y, z = robot.x, robot.y, robot.z
    for _ in range(steps):
        # 更新位置
        x += vx * robot.dt + 0.5 * ax * robot.dt**2
        y += vy * robot.dt + 0.5 * ay * robot.dt**2
        z += vz * robot.dt + 0.5 * az * robot.dt**2
        
        # 更新速度
        vx += ax * robot.dt
        vy += ay * robot.dt
        vz += az * robot.dt
        
        # 保存轨迹
        trajectory.append((x, y, z))
    return np.array(trajectory)

def cal_dynamic_window_vel( state, obstacle):
    """速度采样,得到速度空间窗口 (3D)

    Args:
        v (list): 当前速度 [vx, vy, vz,]
        state (list): 当前机器人状态 [x, y, z,  vx, vy, vz,]
        obstacle (np.ndarray): 障碍物位置，形状为 (N, 3)
        
    Returns:
        list: 最终采样后的速度空间 [vx_low, vx_high, vy_low, vy_high, vz_low, vz_high]
    """
    # 计算速度边界限制

    # 计算三种限制
    Vm = __cal_vel_limit(state)  # 速度边界限制
    Vd = __cal_accel_limit(state)  # 加速度限制
    Va = __cal_obstacle_limit(state, obstacle)  # 障碍物限制

    # 取三种限制的交集范围
    vx_low = max(Vm[0], Vd[0], Va[0])
    vx_high = min(Vm[1], Vd[1], Va[1])
    vy_low = max(Vm[2], Vd[2], Va[2])
    vy_high = min(Vm[3], Vd[3], Va[3])
    vz_low = max(Vm[4], Vd[4], Va[4])
    vz_high = min(Vm[5], Vd[5], Va[5])
  

    return [vx_low, vx_high, vy_low, vy_high, vz_low, vz_high]


def __cal_vel_limit(state):
    """计算速度边界限制 Vm

    Returns:
        list: 速度边界限制 [vx_min, vx_max, vy_min, vy_max, vz_min, vz_max, omega_min, omega_max]
    """
    return [state.v_min, state.v_max, state.v_min, state.v_max, state.v_min, state.v_max]


def __cal_accel_limit(state):
    """计算加速度限制 Vd

    Args:
        vx (float): 当前 x 方向速度
        vy (float): 当前 y 方向速度
        vz (float): 当前 z 方向速度
        omega_z (float): 当前绕 z 轴角速度
        
    Returns:
        list: 考虑加速度时的速度空间 Vd
    """
    vx_low = state.vx - state.a_max * state.dt
    vx_high = state.vx + state.a_max * state.dt
    vy_low = state.vy - state.a_max * state.dt
    vy_high = state.vy + state.a_max * state.dt
    vz_low = state.vz - state.a_max * state.dt    
    vz_high = state.vz + state.a_max * state.dt

    return [vx_low, vx_high, vy_low, vy_high, vz_low, vz_high]


def __cal_obstacle_limit(state, combined_obstacles):
    """环境障碍物限制 Va

    Args:
        state (list): 当前机器人状态 [x, y, z, yaw, vx, vy, vz, omega_z]
        combined_obstacles (list): 混合障碍物列表，包括静态和动态障碍物。
                                   静态障碍物为 (type, x, y, z, radius, height)
                                   动态障碍物为 (type, x, y, z, radius, height)

    Returns:
        list: 考虑障碍物限制的速度空间 Va
    """
    # 提取当前机器人位置
    x, y, z = state.x, state.y, state.z
    # 初始化最近距离为一个较大的值
    min_dist = float('inf')

    # 遍历所有障碍物，找到最近距离
    for obs in combined_obstacles:
        obs_type, obs_x, obs_y, obs_z, radius, height = obs

        if obs_type == 'sphere':
            # 计算与球形障碍物的几何距离
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2 + (z - obs_z)**2) - radius

        elif obs_type == 'cylinder':
            # 计算与圆柱障碍物的几何距离
            xy_dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2) - radius
            z_dist = max(0, abs(z - obs_z) - height / 2)  # Z方向距离
            dist = max(xy_dist, z_dist)

        # 更新最小距离
        if dist < min_dist:
            min_dist = dist

    # 根据最近障碍物距离限制速度
    vx_low = state.v_min
    vx_high = min(state.v_max, np.sqrt(2 * max(min_dist, 0) * state.a_max))
    vy_low = state.v_min
    vy_high = min(state.v_max, np.sqrt(2 * max(min_dist, 0) * state.a_max))
    vz_low = state.v_min
    vz_high = min(state.v_max, np.sqrt(2 * max(min_dist, 0) * state.a_max))
   

    return [vx_low, vx_high, vy_low, vy_high, vz_low, vz_high]







# Dynamic Window Approach
def dynamic_window_approach(robot, goal, obstacles, safe_distance):
    best_trajectory = None
    best_score = 0
    best_velocity = (0, 0, 0)
    [vx_low, vx_high, vy_low, vy_high, vz_low, vz_high]=cal_dynamic_window_vel(robot,obstacles)

    # Sample velocities
    for vx in np.linspace(vx_low, vx_high, 20):
        for vy in np.linspace(vy_low, vy_high, 20):
            for vz in np.linspace(vz_low, vz_high, 10):
                trajectory = predict_trajectory(robot, vx, vy, vz)
                # trajectory=interpolate_trajectory(trajectory, 50)
                score = evaluate_trajectory(trajectory, goal, obstacles,robot, safe_distance)
                if score >best_score:  # 更新最优轨迹
                    best_score = score
                    best_trajectory = trajectory
                    best_velocity = (vx, vy, vz)

    return best_trajectory, best_velocity


# Simulate UAV movement with obstacle avoidance
def simulate_dwa_movement(robot, goal, obstacles, safe_distance, steps=100):
    trajectories = []
    for step in range(steps):
        print(f"Step {step + 1}: Robot position: ({robot.x:.2f}, {robot.y:.2f}, {robot.z:.2f})")

        # Check if goal is reached
        if np.linalg.norm(np.array((robot.x, robot.y, robot.z)) - np.array(goal)) < safe_distance:
            print("Goal reached!")
            break

        # Use DWA to find best trajectory
        trajectory, velocity = dynamic_window_approach(robot, goal, obstacles, safe_distance)
        if trajectory is None:
            print("No valid trajectory found! UAV stopped.")
            break

        trajectories.append(trajectory)
        interpolate_trajectory(trajectories, 50)
        
        # Update robot state
        robot.x, robot.y, robot.z = trajectory[1]
        robot.vx, robot.vy, robot.vz = velocity

    return trajectories


# Visualization
def visualize_simulation(trajectories, obstacles, goal, space_size):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_zlim(0, space_size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("DWA Simulation")

    # Plot obstacles
    for obs in obstacles:
        obs_type, x, y, z, radius, height = obs
        if obs_type == 'sphere':
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            xs = x + radius * np.cos(u) * np.sin(v)
            ys = y + radius * np.sin(u) * np.sin(v)
            zs = z + radius * np.cos(v)
            ax.plot_surface(xs, ys, zs, color='r', alpha=0.6)
        elif obs_type == 'cylinder':
            z_cyl = np.linspace(0, height, 20)
            theta = np.linspace(0, 2 * np.pi, 30)
            theta_grid, z_grid = np.meshgrid(theta, z_cyl)
            x_cyl = x + radius * np.cos(theta_grid)
            y_cyl = y + radius * np.sin(theta_grid)
            ax.plot_surface(x_cyl, y_cyl, z_grid, color='b', alpha=0.6)

    # Plot goal
    ax.scatter(*goal, color='g', s=100, label='Goal')

    # Plot trajectories
    for traj in trajectories:
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='black')

    plt.legend()
    plt.show()





def simulate_dwa_movement_dynamic(robot, goal, obstacles, safe_distance, space_size, steps=100):
    trajectories = []
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_zlim(0, space_size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("DWA Simulation (Dynamic)")

    # Plot obstacles once (static)
    for obs in obstacles:
        obs_type, x, y, z, radius, height = obs
        if obs_type == 'sphere':
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            xs = x + radius * np.cos(u) * np.sin(v)
            ys = y + radius * np.sin(u) * np.sin(v)
            zs = z + radius * np.cos(v)
            ax.plot_surface(xs, ys, zs, color='r', alpha=0.6)
        elif obs_type == 'cylinder':
            z_cyl = np.linspace(0, height, 20)
            theta = np.linspace(0, 2 * np.pi, 30)
            theta_grid, z_grid = np.meshgrid(theta, z_cyl)
            x_cyl = x + radius * np.cos(theta_grid)
            y_cyl = y + radius * np.sin(theta_grid)
            ax.plot_surface(x_cyl, y_cyl, z_grid, color='b', alpha=0.6)

    # Plot goal (static)
    ax.scatter(*goal, color='g', s=100, label='Goal')

    # Initialize UAV trajectory and current position plots
    trajectory_plot, = ax.plot([], [], [], '-k', label='UAV Trajectory')  # UAV trajectory
    current_position_plot, = ax.plot([], [], [], 'ro', label='Current Position')  # Current UAV position

    def update(frame):
        nonlocal robot, trajectories
        print(f"Step {frame + 1}: Robot position: ({robot.x:.2f}, {robot.y:.2f}, {robot.z:.2f})")

        # Check if goal is reached
        if np.linalg.norm(np.array((robot.x, robot.y, robot.z)) - np.array(goal)) < safe_distance:
            print("Goal reached!")
            return trajectory_plot, current_position_plot

        # Use DWA to find best trajectory
        trajectory, velocity = dynamic_window_approach(robot, goal, obstacles, safe_distance)
        if trajectory is None:
            print("No valid trajectory found! UAV stopped.")
            return trajectory_plot, current_position_plot

        # Append the selected trajectory to the full trajectory list
        trajectories.append(trajectory)

        # Update robot state
        robot.x, robot.y, robot.z = trajectory[1]
        robot.vx, robot.vy, robot.vz = velocity

        # Combine all past trajectories into a single array for plotting
        full_trajectory = np.concatenate(trajectories, axis=0)

        # Update the trajectory plot
        trajectory_plot.set_data(full_trajectory[:, 0], full_trajectory[:, 1])
        trajectory_plot.set_3d_properties(full_trajectory[:, 2])

        # Update the current position plot
        current_position_plot.set_data(robot.x, robot.y)
        current_position_plot.set_3d_properties(robot.z)

        return trajectory_plot, current_position_plot

    # Animate the movement
    anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)

    plt.legend()
    plt.show()

    return trajectories


# # Parameters
# space_size = 20
# goal = (5, 6, 8)
# safe_distance = 1.5
# robot = Robot(x=0, y=0, z=0, vx=0, vy=0, vz=0, v_max=1.0, a_max=0.5, dt=0.1)

# # Generate obstacles
# obstacles = generate_mixed_obstacles(30, space_size, radius_range=(0, 1), height_range=(1, 10))

# # Run simulation
# trajectories = simulate_dwa_movement(robot, goal, obstacles, safe_distance)

# # Visualize the simulation
# visualize_simulation(trajectories, obstacles, goal, space_size)
# Parameters

class DynamicObstacle:
    def __init__(self, x, y, z, radius, vx, vy, vz, bounds):
        """
        动态障碍物初始化

        Args:
            x, y, z: 初始位置
            radius: 障碍物半径
            vx, vy, vz: 初始速度
            bounds: 空间范围 (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.bounds = bounds  # 定义障碍物活动范围

    def update_position(self, dt):
        """
        根据速度更新障碍物位置，若超出边界则反弹

        Args:
            dt: 时间步长
        """
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # 检查边界条件，若超出边界则反弹
        if not (self.bounds[0] <= self.x <= self.bounds[1]):
            self.vx *= -1
            self.x = max(self.bounds[0], min(self.x, self.bounds[1]))
        if not (self.bounds[2] <= self.y <= self.bounds[3]):
            self.vy *= -1
            self.y = max(self.bounds[2], min(self.y, self.bounds[3]))
        if not (self.bounds[4] <= self.z <= self.bounds[5]):
            self.vz *= -1
            self.z = max(self.bounds[4], min(self.z, self.bounds[5]))

    def get_position(self):
        """
        获取动态障碍物当前位置及半径，补充为统一的结构

        Returns:
            (type, x, y, z, radius, height)
        """
        return ('dynamic', self.x, self.y, self.z, self.radius, None)

# from matplotlib.animation import FuncAnimation

# def simulate_dwa_movement_with_dynamic_obstacles(robot, goal, dynamic_obstacles, safe_distance, space_size, steps=100):
#     trajectories = []
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim(0, space_size)
#     ax.set_ylim(0, space_size)
#     ax.set_zlim(0, space_size)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title("DWA Simulation with Dynamic Obstacles")

#     # Plot goal (static)
#     ax.scatter(*goal, color='g', s=100, label='Goal')

#     # Initialize UAV trajectory and current position plots
#     trajectory_plot, = ax.plot([], [], [], '-k', label='UAV Trajectory')  # UAV trajectory
#     current_position_plot, = ax.plot([], [], [], 'ro', label='Current Position')  # Current UAV position

#     # Plot dynamic obstacles
#     dynamic_obstacle_plots = []
#     for obstacle in dynamic_obstacles:
#         obstacle_plot, = ax.plot([], [], [], 'bo')  # Dynamic obstacles as blue points
#         dynamic_obstacle_plots.append(obstacle_plot)

#     def update(frame):
#         nonlocal robot, trajectories

#         # Update dynamic obstacles
#         for i, obstacle in enumerate(dynamic_obstacles):
#             obstacle.update_position(robot.dt)
#             obs_type, x, y, z, radius, _ = obstacle.get_position()
#             dynamic_obstacle_plots[i].set_data([x], [y])  # Ensure data is a list
#             dynamic_obstacle_plots[i].set_3d_properties([z])

#         # Convert dynamic obstacles to a list of positions and radii
#         obstacles = [obs.get_position() for obs in dynamic_obstacles]

#         # Check if goal is reached
#         if np.linalg.norm(np.array((robot.x, robot.y, robot.z)) - np.array(goal)) < safe_distance:
#             print("Goal reached!")
#             return trajectory_plot, current_position_plot, *dynamic_obstacle_plots

#         # Use DWA to find best trajectory
#         trajectory, velocity = dynamic_window_approach(robot, goal, obstacles, safe_distance)
#         if trajectory is None:
#             print("No valid trajectory found! UAV stopped.")
#             return trajectory_plot, current_position_plot, *dynamic_obstacle_plots

#         # Append the selected trajectory to the full trajectory list
#         trajectories.append(trajectory)

#         # Update robot state
#         robot.x, robot.y, robot.z = trajectory[1]
#         robot.vx, robot.vy, robot.vz = velocity

#         # Combine all past trajectories into a single array for plotting
#         full_trajectory = np.concatenate(trajectories, axis=0)

#         # Update the trajectory plot
#         trajectory_plot.set_data(full_trajectory[:, 0], full_trajectory[:, 1])
#         trajectory_plot.set_3d_properties(full_trajectory[:, 2])

#         # Update the current position plot
#         current_position_plot.set_data([robot.x], [robot.y])
#         current_position_plot.set_3d_properties([robot.z])

#         return trajectory_plot, current_position_plot, *dynamic_obstacle_plots

#     # Create animation
#     anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)

#     plt.legend()
#     plt.show()

#     return trajectories
def simulate_dwa_movement_with_dynamic_obstacles(robot, goal, dynamic_obstacles, safe_distance, space_size, steps=100):
    trajectories = []
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_zlim(0, space_size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("DWA Simulation with Dynamic Obstacles")

    # Plot goal (static)
    ax.scatter(*goal, color='g', s=100, label='Goal')

    # Initialize UAV trajectory and current position plots
    trajectory_plot, = ax.plot([], [], [], '-k', label='UAV Trajectory')  # UAV trajectory
    current_position_plot, = ax.plot([], [], [], 'ro', label='Current Position')  # Current UAV position

    # Plot dynamic obstacles with varying sizes
    dynamic_obstacle_plots = []
    for obstacle in dynamic_obstacles:
        obs_type, x, y, z, radius, _ = obstacle.get_position()  # 获取初始位置和半径
        obstacle_plot = ax.scatter([x], [y], [z], s=radius * 100, c='b', alpha=0.8)  # 大小由半径决定
        dynamic_obstacle_plots.append(obstacle_plot)

    def update(frame):
        nonlocal robot, trajectories

        # Update dynamic obstacles
        for i, obstacle in enumerate(dynamic_obstacles):
            obstacle.update_position(robot.dt)
            obs_type, x, y, z, radius, _ = obstacle.get_position()
            dynamic_obstacle_plots[i]._offsets3d = ([x], [y], [z])  # 更新位置

        # Convert dynamic obstacles to a list of positions and radii
        obstacles = [obs.get_position() for obs in dynamic_obstacles]

        # Check if goal is reached
        if np.linalg.norm(np.array((robot.x, robot.y, robot.z)) - np.array(goal)) < safe_distance:
            print("Goal reached!")
            return trajectory_plot, current_position_plot, *dynamic_obstacle_plots

        # Use DWA to find best trajectory
        trajectory, velocity = dynamic_window_approach(robot, goal, obstacles, safe_distance)
        if trajectory is None:
            print("No valid trajectory found! UAV stopped.")
            return trajectory_plot, current_position_plot, *dynamic_obstacle_plots

        # Append the selected trajectory to the full trajectory list
        trajectories.append(trajectory)

        # Update robot state
        robot.x, robot.y, robot.z = trajectory[1]
        robot.vx, robot.vy, robot.vz = velocity

        # Combine all past trajectories into a single array for plotting
        full_trajectory = np.concatenate(trajectories, axis=0)

        # Update the trajectory plot
        trajectory_plot.set_data(full_trajectory[:, 0], full_trajectory[:, 1])
        trajectory_plot.set_3d_properties(full_trajectory[:, 2])

        # Update the current position plot
        current_position_plot.set_data([robot.x], [robot.y])
        current_position_plot.set_3d_properties([robot.z])

        return trajectory_plot, current_position_plot, *dynamic_obstacle_plots

    # Create animation
    anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)

    plt.legend()
    plt.show()

    return trajectories


def generate_dynamic_obstacles(num_obstacles, space_size, radius_range, velocity_range):
    dynamic_obstacles = []
    for _ in range(num_obstacles):
        x = np.random.uniform(0, space_size)
        y = np.random.uniform(0, space_size)
        z = np.random.uniform(0, space_size)
        radius = np.random.uniform(radius_range[0], radius_range[1])
        vx = np.random.uniform(velocity_range[0], velocity_range[1])
        vy = np.random.uniform(velocity_range[0], velocity_range[1])
        vz = np.random.uniform(velocity_range[0], velocity_range[1])
        bounds = (0, space_size, 0, space_size, 0, space_size)
        dynamic_obstacles.append(DynamicObstacle(x, y, z, radius, vx, vy, vz, bounds))
    return dynamic_obstacles

# space_size = 20
# goal = (10, 12, 8)
# safe_distance = 1.5
# robot = Robot(x=0, y=0, z=0, vx=0, vy=0, vz=0, v_max=1.0, a_max=0.5, dt=0.1)

# # Generate obstacles
# obstacles = generate_mixed_obstacles(30, space_size, radius_range=(0, 1), height_range=(1, 10))

# # Run simulation with dynamic visualization
# trajectories = simulate_dwa_movement_dynamic(robot, goal, obstacles, safe_distance, space_size)
# Parameters

# class MovingGoal:
#     def __init__(self, x, y, z, vx, vy, vz, bounds):
#         """
#         Initialize the moving goal.

#         Args:
#             x, y, z: Initial position of the goal.
#             vx, vy, vz: Velocity of the goal.
#             bounds: Movement boundaries (x_min, x_max, y_min, y_max, z_min, z_max).
#         """
#         self.x = x
#         self.y = y
#         self.z = z
#         self.vx = vx
#         self.vy = vy
#         self.vz = vz
#         self.bounds = bounds

#     def update_position(self, dt):
#         """
#         Update the position of the goal based on velocity.

#         Args:
#             dt: Time step.
#         """
#         self.x += self.vx * dt
#         self.y += self.vy * dt
#         self.z += self.vz * dt

#         # Ensure the goal stays within bounds
#         if not (self.bounds[0] <= self.x <= self.bounds[1]):
#             self.vx *= -1
#             self.x = max(self.bounds[0], min(self.x, self.bounds[1]))
#         if not (self.bounds[2] <= self.y <= self.bounds[3]):
#             self.vy *= -1
#             self.y = max(self.bounds[2], min(self.y, self.bounds[3]))
#         if not (self.bounds[4] <= self.z <= self.bounds[5]):
#             self.vz *= -1
#             self.z = max(self.bounds[4], min(self.z, self.bounds[5]))

#     def get_position(self):
#         """
#         Get the current position of the goal.

#         Returns:
#             (x, y, z): Current position of the goal.
#         """
#         return (self.x, self.y, self.z)
import numpy as np

class MovingGoal:
    def __init__(self, x, y, z, bounds, v_max=1.0, direction_change_interval=10):
        """
        Initialize a structured random moving goal.

        Args:
            x, y, z: Initial position.
            bounds: Movement boundaries (x_min, x_max, y_min, y_max, z_min, z_max).
            v_max: Maximum velocity.
            direction_change_interval: Steps after which direction is recalculated.
        """
        self.x = x
        self.y = y
        self.z = z
        self.bounds = bounds
        self.v_max = v_max
        self.direction_change_interval = direction_change_interval
        self.step_count = 0

        # Initialize random direction
        self.vx, self.vy, self.vz = self._generate_random_direction()

    def _generate_random_direction(self):
        """
        Generate a random direction vector with fixed magnitude.
        """
        direction = np.random.randn(3)  # Random vector
        direction = direction / np.linalg.norm(direction) * self.v_max  # Normalize and scale
        return direction[0], direction[1], direction[2]

    def update_position(self, dt):
        """
        Update position using a structured random movement.

        Args:
            dt: Time step.
        """
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Keep within bounds and bounce back if hitting boundaries
        if not (self.bounds[0] <= self.x <= self.bounds[1]):
            self.vx *= -1
            self.x = np.clip(self.x, self.bounds[0], self.bounds[1])
        if not (self.bounds[2] <= self.y <= self.bounds[3]):
            self.vy *= -1
            self.y = np.clip(self.y, self.bounds[2], self.bounds[3])
        if not (self.bounds[4] <= self.z <= self.bounds[5]):
            self.vz *= -1
            self.z = np.clip(self.z, self.bounds[4], self.bounds[5])

        # Change direction periodically
        self.step_count += 1
        if self.step_count >= self.direction_change_interval:
            self.vx, self.vy, self.vz = self._generate_random_direction()
            self.step_count = 0

    def get_position(self):
        """
        Get the current position of the goal.

        Returns:
            (x, y, z): Current position of the goal.
        """
        return (self.x, self.y, self.z)



# space_size = 20
# goal = (10, 12, 15)
# safe_distance = 0.1
# robot = Robot(x=0, y=0, z=0, vx=0, vy=0, vz=0, v_max=1.0, a_max=0.5, dt=0.1)

# # Generate dynamic obstacles
# dynamic_obstacles = generate_dynamic_obstacles(30, space_size, radius_range=(3, 5), velocity_range=(-5, 5))

# # Run simulation with dynamic obstacles
# trajectories = simulate_dwa_movement_with_dynamic_obstacles(robot, goal, dynamic_obstacles, safe_distance, space_size)
# Parameters
# def simulate_dwa_movement_with_dynamic_obstacles_and_moving_goal(robot, moving_goal, dynamic_obstacles, safe_distance, space_size, steps=100):
#     trajectories = []
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim(0, space_size)
#     ax.set_ylim(0, space_size)
#     ax.set_zlim(0, space_size)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title("DWA Simulation with Dynamic Obstacles and Moving Goal")

#     # Initialize UAV trajectory and current position plots
#     trajectory_plot, = ax.plot([], [], [], '-k', label='UAV Trajectory')  # UAV trajectory
#     current_position_plot, = ax.plot([], [], [], 'ro', label='Current Position')  # Current UAV position

#     # Plot dynamic obstacles
#     dynamic_obstacle_plots = []
#     for obstacle in dynamic_obstacles:
#         obs_type, x, y, z, radius, _ = obstacle.get_position()
#         obstacle_plot = ax.scatter([x], [y], [z], s=radius * 100, c='b', alpha=0.8)
#         dynamic_obstacle_plots.append(obstacle_plot)

#     # Initialize moving goal
#     goal_plot = ax.scatter(*moving_goal.get_position(), color='g', s=100, label='Moving Goal')

#     def update(frame):
#         nonlocal robot, trajectories

#         # Update dynamic obstacles
#         for i, obstacle in enumerate(dynamic_obstacles):
#             obstacle.update_position(robot.dt)
#             obs_type, x, y, z, radius, _ = obstacle.get_position()
#             dynamic_obstacle_plots[i]._offsets3d = ([x], [y], [z])

#         # Update the moving goal
#         moving_goal.update_position(robot.dt)
#         goal_position = moving_goal.get_position()
#         goal_plot._offsets3d = ([goal_position[0]], [goal_position[1]], [goal_position[2]])

#         # Convert dynamic obstacles to a list of positions and radii
#         obstacles = [obs.get_position() for obs in dynamic_obstacles]

#         # Check if goal is reached
#         if np.linalg.norm(np.array((robot.x, robot.y, robot.z)) - np.array(goal_position)) < safe_distance:
#             print("Goal reached!")
#             return trajectory_plot, current_position_plot, *dynamic_obstacle_plots, goal_plot

#         # Use DWA to find best trajectory
#         trajectory, velocity = dynamic_window_approach(robot, goal_position, obstacles, safe_distance)
#         if trajectory is None:
#             print("No valid trajectory found! UAV stopped.")
#             return trajectory_plot, current_position_plot, *dynamic_obstacle_plots, goal_plot

#         # Append the selected trajectory to the full trajectory list
#         trajectories.append(trajectory)

#         # Update robot state
#         robot.x, robot.y, robot.z = trajectory[1]
#         robot.vx, robot.vy, robot.vz = velocity

#         # Combine all past trajectories into a single array for plotting
#         full_trajectory = np.concatenate(trajectories, axis=0)

#         # Update the trajectory plot
#         trajectory_plot.set_data(full_trajectory[:, 0], full_trajectory[:, 1])
#         trajectory_plot.set_3d_properties(full_trajectory[:, 2])

#         # Update the current position plot
#         current_position_plot.set_data([robot.x], [robot.y])
#         current_position_plot.set_3d_properties([robot.z])

#         return trajectory_plot, current_position_plot, *dynamic_obstacle_plots, goal_plot

#     # Create animation
#     anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)

#     plt.legend()
#     plt.show()

#     return trajectories

# space_size = 20
# safe_distance = 1.5
# robot = Robot(x=0, y=0, z=0, vx=0, vy=0, vz=0, v_max=1.0, a_max=0.5, dt=0.1)

# # Generate dynamic obstacles
# dynamic_obstacles = generate_dynamic_obstacles(20, space_size, radius_range=(1, 5), velocity_range=(-2, 2))
# # Define movement boundaries (x_min, x_max, y_min, y_max, z_min, z_max)
# bounds = (0, 20, 0, 20, 0, 20)

# # Create a moving goal
# moving_goal = MovingGoal(x=10, y=10, z=10, bounds=bounds, v_max=15.0, direction_change_interval=50)
# # # Create moving goal
# # moving_goal = MovingGoal(x=15, y=15, z=15, vx=-1, vy=-1, vz=-1, bounds=(0, space_size, 0, space_size, 0, space_size))

# # Run simulation
# trajectories = simulate_dwa_movement_with_dynamic_obstacles_and_moving_goal(robot, moving_goal, dynamic_obstacles, safe_distance, space_size)
class DynamicObstacle:
    def __init__(self, x, y, z, vx, vy, vz, bounds, v_max=0.5, direction_change_interval=50):
        """
        Initialize a dynamic obstacle with velocity and bounds.

        Args:
            x, y, z: Initial position.
            vx, vy, vz: Initial velocity.
            bounds: Movement boundaries (x_min, x_max, y_min, y_max, z_min, z_max).
            v_max: Maximum velocity.
            direction_change_interval: Steps after which direction is recalculated.
        """
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.bounds = bounds
        self.v_max = v_max
        self.direction_change_interval = direction_change_interval
        self.step_count = 0

    def _generate_random_direction(self):
        """Generate a random direction vector with fixed magnitude."""
        direction = np.random.randn(3)  # Random vector
        direction = direction / np.linalg.norm(direction) * self.v_max  # Normalize and scale
        return direction[0], direction[1], direction[2]

    def update_position(self, dt):
        """
        Update position using a structured random movement.

        Args:
            dt: Time step.
        """
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Bounce back on boundary collision
        if not (self.bounds[0] <= self.x <= self.bounds[1]):
            self.vx *= -1
            self.x = np.clip(self.x, self.bounds[0], self.bounds[1])
        if not (self.bounds[2] <= self.y <= self.bounds[3]):
            self.vy *= -1
            self.y = np.clip(self.y, self.bounds[2], self.bounds[3])
        if not (self.bounds[4] <= self.z <= self.bounds[5]):
            self.vz *= -1
            self.z = np.clip(self.z, self.bounds[4], self.bounds[5])

        # Periodically change direction
        self.step_count += 1
        if self.step_count >= self.direction_change_interval:
            self.vx, self.vy, self.vz = self._generate_random_direction()
            self.step_count = 0

    def get_position(self):
        """Return the current position and radius of the obstacle."""
        return ('sphere', self.x, self.y, self.z, 1.0, None)  # Assuming dynamic obstacles are spheres

def simulate_with_obstacles(robot, goal, static_obstacles, dynamic_obstacles, safe_distance, space_size, steps=100):
    # Combine static and dynamic obstacles
    combined_obstacles = static_obstacles + [obs.get_position() for obs in dynamic_obstacles]

    # Continue the simulation logic as before
    trajectories = []

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_zlim(0, space_size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("DWA Simulation with Static and Dynamic Obstacles")

    # Plot goal
    goal_plot = ax.scatter(*goal.get_position(), color='g', s=100, label='Moving Goal')

    # Plot static obstacles
    for obs in static_obstacles:
        obs_type, x, y, z, radius, height = obs  # Unpack all 6 values
        if obs_type == 'sphere':
            # Plot a sphere
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            xs = x + radius * np.cos(u) * np.sin(v)
            ys = y + radius * np.sin(u) * np.sin(v)
            zs = z + radius * np.cos(v)
            ax.plot_surface(xs, ys, zs, color='r', alpha=0.6)
        elif obs_type == 'cylinder':
            # Plot a cylinder
            z_cyl = np.linspace(0, height, 20)
            theta = np.linspace(0, 2 * np.pi, 30)
            theta_grid, z_grid = np.meshgrid(theta, z_cyl)
            x_cyl = x + radius * np.cos(theta_grid)
            y_cyl = y + radius * np.sin(theta_grid)
            ax.plot_surface(x_cyl, y_cyl, z_grid, color='r', alpha=0.6)


    # Plot dynamic obstacles
    dynamic_obstacle_plots = []
    for obstacle in dynamic_obstacles:
        obstacle_plot, = ax.plot([], [], [], 'bo', label='Dynamic Obstacle')
        dynamic_obstacle_plots.append(obstacle_plot)
        # Plot dynamic obstacles


    # Plot UAV trajectory and current position
    trajectory_plot, = ax.plot([], [], [], '-k', label='UAV Trajectory')
    current_position_plot, = ax.plot([], [], [], 'ro', label='Current Position')

    def update(frame):
        nonlocal robot, trajectories

        # Update dynamic obstacles
        for i, obstacle in enumerate(dynamic_obstacles):
            obstacle.update_position(robot.dt)
            obs_type, x, y, z, radius, height = obstacle.get_position()
            dynamic_obstacle_plots[i].set_data([x], [y])
            dynamic_obstacle_plots[i].set_3d_properties([z])

        # Combine static and dynamic obstacles for collision checking
        combined_obstacles = static_obstacles + [obs.get_position() for obs in dynamic_obstacles]

        # Check if goal is reached
        if np.linalg.norm(np.array(robot.get_state()[:3]) - np.array(goal.get_position())) < safe_distance:
            print("Goal reached!")
            return trajectory_plot, current_position_plot, *dynamic_obstacle_plots

        # Use DWA to find the best trajectory
        trajectory, velocity = dynamic_window_approach(robot, goal.get_position(), combined_obstacles, safe_distance)
        if trajectory is None:
            print("No valid trajectory found! UAV stopped.")
            return trajectory_plot, current_position_plot, *dynamic_obstacle_plots

        # Append trajectory and update robot state
        trajectories.append(trajectory)
        robot.update_state(trajectory[1], velocity)

        # Combine all past trajectories for plotting
        full_trajectory = np.concatenate(trajectories, axis=0)

        # Update trajectory plot
        trajectory_plot.set_data(full_trajectory[:, 0], full_trajectory[:, 1])
        trajectory_plot.set_3d_properties(full_trajectory[:, 2])

        # Update current position plot
        current_position_plot.set_data([robot.x], [robot.y])
        current_position_plot.set_3d_properties([robot.z])

        # Update goal position
        goal.update_position(robot.dt)
        goal_position = goal.get_position()
        goal_plot._offsets3d = ([goal_position[0]], [goal_position[1]], [goal_position[2]])

        return trajectory_plot, current_position_plot, *dynamic_obstacle_plots

    # Run animation
    anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)

    plt.legend()
    plt.show()

    return trajectories
# Define space boundaries (x_min, x_max, y_min, y_max, z_min, z_max)
space_size = 20
bounds = (0, space_size, 0, space_size, 0, space_size)

# Create the UAV (robot)
robot = Robot(x=1, y=1, z=10, vx=0, vy=0, vz=0, v_max=1.0, v_min=-1.5,a_max=1.0, a_min=-0.5, dt=0.1)

# Create the moving goal
goal = MovingGoal(x=18, y=18, z=18, bounds=bounds, v_max=0, direction_change_interval=50)

# Define static obstacles with consistent format
# Define static obstacles
static_obstacles = [
    # ('sphere', 5, 5, 5, 1.5, None),  # Sphere at (5, 5, 5) with radius 1.5
    # ('sphere', 10, 10, 10, 2.0, None),  # Sphere at (10, 10, 10) with radius 2.0
    ('cylinder', 15, 15, 0, 1.0, 20.0) ,  # Cylinder at (15, 15) with radius 1.0 and height 5.0
     ('cylinder', 3, 3, 0, 1.0,20.0),  # Cylinder at (2, 2) with radius 1.0 and height 10.0
    ('cylinder', 6, 15, 0, 1.0, 20.0) ,  # Cylinder at (15, 15) with radius 1.0 and height 5.0
     ('cylinder', 10, 3, 0, 1.0,20.0) , # Cylinder at (2, 2) with radius 1.0 and height 10.0
    ('cylinder', 7, 3, 0, 1.5,20.0)  ,# Cylinder at (2, 2) with radius 1.0 and height 10.0

    ('cylinder', 5, 10, 0, 1.5,20.0),  # Cylinder at (2, 2) with radius 1.0 and height 10.0
      ('cylinder', 10, 15, 0, 1.5,20.0)  # Cylinder at (2, 2) with radius 1.0 and height 10.0


     ]


# Create dynamic obstacles
dynamic_obstacles = [
    DynamicObstacle(x=18, y=18, z=18, vx=0.2, vy=0.2, vz=0.1, bounds=bounds, v_max=5),
    DynamicObstacle(x=15, y=16, z=18, vx=-0.1, vy=0.2, vz=0.15, bounds=bounds, v_max=5)
]

# Define the safety distance
safe_distance = 1

# Run the simulation
trajectories = simulate_with_obstacles(robot, goal, static_obstacles, dynamic_obstacles, safe_distance, space_size, steps=200)

# Visualize results
print("Simulation completed.")
