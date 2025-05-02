# functions.py
import cv2
import numpy as np

def calibrate_single_camera(image_paths, chessboard_size=(7, 9), square_size=15):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if not objpoints:
        raise ValueError("No valid chessboard corners found. Check your images.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    reprojection_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reprojection_error += error

    reprojection_error /= len(objpoints)

    return K, dist, reprojection_error

def rectify_images(img1, img2, K, dist, K1, dist1, T):
    R = T[:3, :3]
    t = T[:3, 3]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist,
        K1, dist1,
        img1.shape[:2][::-1],
        R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(K, dist, R1, P1, img1.shape[:2][::-1], cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K1, dist1, R2, P2, img2.shape[:2][::-1], cv2.CV_32FC1)

    img1_rectified = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)

    return img1_rectified, img2_rectified, Q

# functions.py (add this to your functions file)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def dense_3d_reconstruction(Q, output_ply_path="output.ply"):
    left_img_color = cv2.imread("rectified_img1.png")  # Load in color (BGR)
    left_img = cv2.cvtColor(left_img_color, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    right_img = cv2.imread("rectified_img2.png", cv2.IMREAD_GRAYSCALE)
    print(Q)

    # Define StereoSGBM parameters
    min_disparity = 0
    num_disparities = 16*4 # Must be a multiple of 16
    block_size = 11
    uniqueness_ratio = 15
    speckle_window_size = 100
    speckle_range = 32
    disp12_max_diff = 1
    pre_filter_cap = 32

    # Create StereoSGBM object for left and right images
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        disp12MaxDiff=disp12_max_diff,
        P1=  8 * 3 * block_size**2,  # P1: Smoothing penalty
        P2=  32 * 3 * block_size**2,  # P2: Stronger smoothing penalty
        preFilterCap=pre_filter_cap,
        mode=cv2.STEREO_SGBM_MODE_HH # Improves disparity quality
    )

    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)  # Right matcher for WLS

    # Compute disparity maps
    disparity_left = stereo_left.compute(left_img, right_img).astype(np.float32) / 16.0
    disparity_right = stereo_right.compute(right_img, left_img).astype(np.float32) / 16.0

    # Create WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
    wls_filter.setLambda(8000)  # Higher values smooth more
    wls_filter.setSigmaColor(2.5)  # Edge-preserving filter

    # Apply WLS filtering
    filtered_disparity = wls_filter.filter(disparity_left, left_img, disparity_map_right=disparity_right)

    # Normalize disparity for visualization
    filtered_disparity_norm = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    filtered_disparity_norm = np.uint8(filtered_disparity_norm)

    # Show results
    plt.figure(figsize=(10, 5))
    plt.imshow(filtered_disparity_norm, cmap="jet")
    plt.colorbar()
    plt.title("Filtered  Map ")
    plt.show()
    # Reproject to 3D
    points_3D = cv2.reprojectImageTo3D(filtered_disparity, Q)

    # Create a valid mask
    mask = (filtered_disparity > 10) & np.isfinite(points_3D[:, :, 0])

    # Extract valid 3D points and colors
    points = points_3D[mask]
    colors = left_img[mask]
    colors = colors.astype(np.float32) / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save and visualize
    o3d.io.write_point_cloud(output_ply_path, pcd)
    o3d.visualization.draw_geometries([pcd])

    print(f"3D point cloud saved as {output_ply_path}")



import cv2
import numpy as np
import matplotlib.pyplot as plt
def sparse(rectified_img1,rectified_img2,K,img1):
# Feature Detection and Matching using SIFT

    akaze = cv2.AKAZE_create()
    bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    # Detect keypoints and descriptors in both images
    kp1, des1 = akaze.detectAndCompute(rectified_img1, None)
    kp2, des2 = akaze.detectAndCompute(rectified_img2, None)

    # Match bf_akaze using FLANN-based matcher
    matches = bf_akaze.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance

    # Set a threshold to filter out weak matches
    # This threshold can be adjusted based on your needs
    threshold = 100 # Matches with distance < 50 are considered strong
    good_matches = [m for m in matches if m.distance < threshold]

    # Print number of good matches
    print(f"Total matches found: {len(matches)}")
    print(f"Total strong matches after filtering: {len(good_matches)}")

    # Extract the matched keypoints for the good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Use RANSAC to compute the Fundamental Matrix and filter out outliers
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3)

    # Select inliers (good matches after RANSAC)
    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]


    # Visualize the matched features after filtering with RANSAC
    matched_img = cv2.drawMatches(rectified_img1, kp1, rectified_img2, kp2, good_matches, None, 
                                matchesMask=mask.ravel().tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    E = K.T @ F @ K
    print("Essential Matrix:")
    print(E)

    # Decompose Essential Matrix to get Rotation and Translation
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Possible solutions
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    print("Rotation Matrices:")
    print(R1)
    print(R2)
    print("Translation Vector:")
    print(t)
    # Display the matched features
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title("Strong Feature Matches")
    plt.axis("off")
    plt.show()

    # # Save the matched features image
    cv2.imwrite("strong_feature_matches_with_ransac.jpg", matched_img)

    U, S, Vt = np.linalg.svd(E)
    print(f"Singular values of E: {S}")

    # Check if the third singular value is (approximately) zero
    if np.isclose(S[2], 0, atol=1e-5):
        print("Essential matrix has the correct rank (2).")
    else:
        print("Essential matrix does NOT have the correct rank!")
    det_E = np.linalg.det(E)
    print(f"Det(E) = {det_E}")
    if np.isclose(det_E, 0, atol=1e-5):
        det_E_string="Essential matrix satisfies det(E) = 0 condition."
        print("Essential matrix satisfies det(E) = 0 condition.")
    else:
        det_E_string="Essential matrix does NOT satisfy det(E) = 0 condition!"
        print("Essential matrix does NOT satisfy det(E) = 0 condition!")


    # Verify rotation matrices
    det_R1 = np.linalg.det(R1)
    det_R2 = np.linalg.det(R2)

    print(f"Det(R1) = {det_R1}, Det(R2) = {det_R2}")
    identity_matrix = np.eye(3)

    # Compute deviations from identity matrix
    R1_deviation = np.linalg.norm(R1 - identity_matrix)
    R2_deviation = np.linalg.norm(R2 - identity_matrix)

    print(f"Deviation of R1 from Identity: {R1_deviation}")
    print(f"Deviation of R2 from Identity: {R2_deviation}")
    errors = []
    for i in range(len(pts1)):
        x1 = np.append(pts1[i], 1)  # Convert to homogeneous coordinates
        x2 = np.append(pts2[i], 1)
        
        error = np.abs(x2.T @ F @ x1)  # Compute epipolar constraint error
        errors.append(error)

    mean_error = np.mean(errors)
    print(f"Mean Epipolar Constraint Error: {mean_error}")
    T=t
    print("Original T shape:", T.shape)
    if T.shape == (3,):
        T = T.reshape(3, 1)
    print("Fixed T shape:", T.shape)

    # Choose the correct rotation matrix (closest to identity)
    identity_matrix = np.eye(3)
    R1_deviation = np.linalg.norm(R1 - identity_matrix)
    R2_deviation = np.linalg.norm(R2 - identity_matrix)

    if R1_deviation < R2_deviation:
        R = R1
        print("Selected R1")
    else:
        R = R2
        print("Selected R2")

    # Compute projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K [I | 0]
    P2 = K @ np.hstack((R, T))  # P2 = K [R | T]

    # Ensure pts1 and pts2 are 2D
    print("pts1 shape before reshape:", pts1.shape)
    print("pts2 shape before reshape:", pts2.shape)

    if pts1.ndim == 3:
        pts1 = pts1.reshape(-1, 2)
    if pts2.ndim == 3:
        pts2 = pts2.reshape(-1, 2)

    # Convert points to homogeneous coordinates
    pts1_hom = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))  # Shape: (3, N)
    pts2_hom = np.vstack((pts2.T, np.ones((1, pts2.shape[0]))))  # Shape: (3, N)

    # Perform triangulation
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_hom[:2], pts2_hom[:2])

    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Shape: (3, N)

    # Filter out points with negative depth (Z should be positive)
    valid_indices = points_3d[2] > 0
    points_3d_filtered = points_3d[:, valid_indices]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K [I | 0]

    # Step 2: Reproject 3D points onto the first image
    points_4d_hom = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))  # Convert to homogeneous coordinates
    reprojected_pts = P1 @ points_4d_hom  # Project to 2D
    reprojected_pts = reprojected_pts[:2] / reprojected_pts[2]  # Normalize to get (u, v) coordinates


    # Step 3: Visualize the reprojected points on the original image
    img1_with_points = img1.copy()  # Make a copy of the original image

    # Draw the reprojected points on the image
    for pt in reprojected_pts.T:
        u, v = int(pt[0]), int(pt[1])  # Convert to integer pixel coordinates
        cv2.circle(img1_with_points, (u, v), radius=5, color=(0, 255, 0), thickness=-1)  # Green points

    # Display the image with reprojected points
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img1_with_points, cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Reprojected 3D Points")
    plt.axis('off')
    plt.show()

    return F, E, R1, R2, t, det_R1, det_R2,det_E_string
    # Compute Essential Matrix
    

