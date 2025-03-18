import numpy as np
import cv2
import ai2thor.controller
import math
import time


class OpticalFlowNavigator:
    def __init__(self, target_position=(0, 0), scene="FloorPlan212"):
        """
        Initialize the navigator with target position and scene.
        
        Args:
            target_position (tuple): Target (x, z) coordinates in AI2-THOR space
            scene (str): AI2-THOR scene name
        """
        # InitializingAI2-THOR controller
        self.controller = ai2thor.controller.Controller(
            visibilityDistance=1.5,
            scene=scene,
            gridSize=0.25,
            moveAmount=0.25,
            rotateAmount=45,
            fieldOfView=60,
            width=640,
            height=480,
            renderDepthImage=True,
        )
        
        # Target position
        self.target_position = target_position
        
        # LK optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Initialize previous frame and features
        self.prev_gray = None
        self.prev_pts = None
        
        # Control parameters
        self.min_distance_to_target = 0.5
        self.max_iterations = 500
        self.repulsion_factor = 0.3
        self.attraction_factor = 0.7
        
        # Initialize metrics
        self.iteration_count = 0
        self.path_length = 0
        self.prev_position = None
        
    def update_position(self):
        """Update the agent's position from the controller."""
        event = self.controller.step(action="Done")
        return event.metadata["agent"]["position"]
    
    def get_distance_to_target(self, position):
        """Calculate distance to target position."""
        tx, tz = self.target_position
        px, _, pz = position["x"], position["y"], position["z"]
        return math.sqrt((tx - px) ** 2 + (tz - pz) ** 2)
    
    def compute_optical_flow(self, frame):
        """
        Compute optical flow between consecutive frames.
        
        Args:
            frame (numpy.ndarray): Current RGB frame from AI2-THOR
        
        Returns:
            tuple: (flow_vectors, flow_status, new_pts)
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Initialize on first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return None, None, self.prev_pts
        
        # Calculate optical flow
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            curr_pts, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )
            
            # Filter valid points
            valid_curr = curr_pts[status == 1]
            valid_prev = self.prev_pts[status == 1]
            
            # Calculate flow vectors
            flow_vectors = valid_curr - valid_prev
            
            # Update previous frame and points
            self.prev_gray = gray
            
            # Refresh features periodically
            if len(valid_prev) < 30:
                self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            else:
                self.prev_pts = valid_curr.reshape(-1, 1, 2)
                
            return flow_vectors, status, valid_curr
        else:
            # If no points to track, refresh features
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return None, None, self.prev_pts
    
    def detect_focus_of_expansion(self, flow_vectors, points):
        """
        Detect Focus of Expansion (FOE) from optical flow vectors.
        
        Args:
            flow_vectors (numpy.ndarray): Optical flow vectors
            points (numpy.ndarray): Current feature points
        
        Returns:
            tuple: (x, y) coordinates of FOE
        """
        if flow_vectors is None or len(flow_vectors) < 5:
            # Not enough vectors to determine FOE
            return None
        
        # Normalize flow vectors
        flow_norms = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
        flow_norms = np.where(flow_norms == 0, 1, flow_norms)  # Avoid division by zero
        normalized_flow = flow_vectors / flow_norms[:, np.newaxis]
        
        # RANSAC to find FOE
        best_foe = None
        best_inliers = 0
        iterations = 50
        threshold = 0.1
        
        for _ in range(iterations):
            if len(points) < 2:
                continue
                
            # Randomly select two points
            indices = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[indices]
            v1, v2 = normalized_flow[indices]
            
            # Solve for FOE (intersection of lines)
            A = np.array([v1, v2])
            b = np.array([
                p1[0] * v1[0] + p1[1] * v1[1],
                p2[0] * v2[0] + p2[1] * v2[1]
            ])
            
            try:
                foe = np.linalg.solve(A, b)
                
                # Count inliers
                inliers = 0
                for p, v in zip(points, normalized_flow):
                    # Vector from point to FOE
                    vec_to_foe = foe - p
                    # Normalize
                    norm = np.linalg.norm(vec_to_foe)
                    if norm > 0:
                        vec_to_foe = vec_to_foe / norm
                        # Dot product with flow vector
                        alignment = np.abs(np.dot(vec_to_foe, v))
                        if alignment > (1 - threshold):
                            inliers += 1
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_foe = foe
            except np.linalg.LinAlgError:
                # Singular matrix, lines are parallel
                continue
        
        return best_foe
    
    def generate_potential_field(self, frame, depth_frame, foe):
        """
        Generate a potential field for navigation based on FOE and depth image.
        
        Args:
            frame (numpy.ndarray): Current RGB frame
            depth_frame (numpy.ndarray): Depth frame
            foe (tuple): Focus of Expansion coordinates
        
        Returns:
            tuple: (attractive_vector, repulsive_vector)
        """
        height, width = frame.shape[:2]
        
        # Default vectors
        attractive_vector = np.array([0, 0])
        repulsive_vector = np.array([0, 0])
        
        # If FOE is detected
        if foe is not None:
            # FOE coordinates
            fx, fy = foe
            
            # Clamp FOE to image boundaries
            fx = max(0, min(width - 1, fx))
            fy = max(0, min(height - 1, fy))
            
            # Calculate attractive vector (towards target)
            center_x, center_y = width // 2, height // 2
            attractive_vector = np.array([center_x - fx, center_y - fy])
            
            # Normalize attractive vector
            norm = np.linalg.norm(attractive_vector)
            if norm > 0:
                attractive_vector = attractive_vector / norm
        
        # Generate repulsive vector from depth image
        if depth_frame is not None:
            # Normalize depth
            depth_norm = cv2.normalize(depth_frame, None, 0, 1, cv2.NORM_MINMAX)
            
            # Create distance weight map (closer objects have higher weight)
            distance_weight = 1 - depth_norm
            
            # Calculate center of mass for close objects
            y_indices, x_indices = np.indices((height, width))
            
            # Threshold to consider only close objects
            threshold = 0.7
            mask = distance_weight > threshold
            
            if np.any(mask):
                # Weighted center of mass
                com_x = np.sum(x_indices * mask) / np.sum(mask)
                com_y = np.sum(y_indices * mask) / np.sum(mask)
                
                # Calculate repulsive vector (away from obstacles)
                center_x, center_y = width // 2, height // 2
                repulsive_vector = np.array([center_x - com_x, center_y - com_y])
                
                # Normalize repulsive vector
                norm = np.linalg.norm(repulsive_vector)
                if norm > 0:
                    repulsive_vector = repulsive_vector / norm
        
        return attractive_vector, repulsive_vector
    
    def determine_action(self, combined_vector):
        """
        Determine the next action based on the combined vector.
        
        Args:
            combined_vector (numpy.ndarray): Combined navigation vector
        
        Returns:
            str: Action to take
        """
        # Convert vector to angle
        angle = math.atan2(combined_vector[1], combined_vector[0])
        angle_degrees = math.degrees(angle)
        
        # Forward: -45 to 45 degrees
        # Right: 45 to 135 degrees
        # Backward: 135 to -135 degrees
        # Left: -135 to -45 degrees
        
        if -45 <= angle_degrees <= 45:
            return "MoveAhead"
        elif 45 < angle_degrees <= 135:
            return "RotateRight"
        elif angle_degrees > 135 or angle_degrees < -135:
            return "RotateLeft"  # Backward case, but we'll rotate instead
        else:  # -135 to -45
            return "RotateLeft"
    
    def run(self):
        """
        Main navigation loop.
        
        Returns:
            dict: Navigation metrics
        """
        # Initialize
        position = self.update_position()
        self.prev_position = position
        distance_to_target = self.get_distance_to_target(position)
        
        # Navigation loop
        while distance_to_target > self.min_distance_to_target and self.iteration_count < self.max_iterations:
            # Get current frame
            event = self.controller.step(action="Done")
            frame = event.frame
            depth_frame = event.depth_frame
            
            # Compute optical flow
            flow_vectors, status, curr_pts = self.compute_optical_flow(frame)
            
            # Detect FOE
            foe = None
            if flow_vectors is not None and len(flow_vectors) > 0:
                foe = self.detect_focus_of_expansion(flow_vectors, curr_pts)
            
            # Generate potential field
            attractive_vector, repulsive_vector = self.generate_potential_field(frame, depth_frame, foe)
            
            # Combine vectors
            combined_vector = (self.attraction_factor * attractive_vector + 
                              self.repulsion_factor * repulsive_vector)
            
            # Normalize combined vector
            norm = np.linalg.norm(combined_vector)
            if norm > 0:
                combined_vector = combined_vector / norm
            
            # Determine and execute action
            action = self.determine_action(combined_vector)
            event = self.controller.step(action=action)
            
            # Update position and distance
            position = event.metadata["agent"]["position"]
            distance_to_target = self.get_distance_to_target(position)
            
            # Update path length
            if self.prev_position:
                dx = position["x"] - self.prev_position["x"]
                dz = position["z"] - self.prev_position["z"]
                self.path_length += math.sqrt(dx**2 + dz**2)
            self.prev_position = position
            
            # Increment iteration count
            self.iteration_count += 1
            
            # Visualize (optional)
            self.visualize(frame, flow_vectors, curr_pts, foe, combined_vector)
            
            # Small delay to see visualization
            cv2.waitKey(1)
        
        # Clean up
        cv2.destroyAllWindows()
        
        # Return metrics
        return {
            "iterations": self.iteration_count,
            "path_length": self.path_length,
            "final_distance": distance_to_target,
            "success": distance_to_target <= self.min_distance_to_target
        }
    
    def visualize(self, frame, flow_vectors, points, foe, combined_vector):
        """
        Visualize the navigation process.
        
        Args:
            frame (numpy.ndarray): Current RGB frame
            flow_vectors (numpy.ndarray): Optical flow vectors
            points (numpy.ndarray): Current feature points
            foe (tuple): Focus of Expansion coordinates
            combined_vector (numpy.ndarray): Combined navigation vector
        """
        # Create a copy for visualization
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Draw flow vectors
        if flow_vectors is not None and points is not None:
            for i, (point, vector) in enumerate(zip(points, flow_vectors)):
                p1 = tuple(point.astype(int))
                p2 = tuple((point + vector * 5).astype(int))
                cv2.arrowedLine(vis_frame, p1, p2, (0, 255, 0), 2)
        
        # Draw FOE
        if foe is not None:
            fx, fy = foe
            if 0 <= fx < width and 0 <= fy < height:
                cv2.circle(vis_frame, (int(fx), int(fy)), 10, (0, 0, 255), -1)
        
        # Draw combined vector
        if combined_vector is not None:
            end_point = (
                int(center_x + combined_vector[0] * 50),
                int(center_y + combined_vector[1] * 50)
            )
            cv2.arrowedLine(vis_frame, (center_x, center_y), end_point, (255, 0, 0), 3)
        
        # Display metrics
        cv2.putText(
            vis_frame,
            f"Distance: {self.get_distance_to_target(self.prev_position):.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Show visualization
        cv2.imshow("AI2-THOR Navigation", vis_frame)


def main():
    """Main function to run the navigation system."""
    # Set target position
    target_position = (1.0, 3.0)  # Example target position
    
    # Initialize and run navigator
    navigator = OpticalFlowNavigator(target_position=target_position, scene="FloorPlan212")
    metrics = navigator.run()
    
    # Print results
    print("Navigation completed!")
    print(f"Iterations: {metrics['iterations']}")
    print(f"Path length: {metrics['path_length']:.2f}")
    print(f"Final distance to target: {metrics['final_distance']:.2f}")
    print(f"Success: {metrics['success']}")
    

if __name__ == "__main__":
    main()
