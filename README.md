This project demonstrates a complete stereo vision pipeline using Python, OpenCV, and Open3D — from camera calibration to 3D point cloud generation.

🔧 Features Implemented
✅ Monocular camera calibration (left & right)
✅ Stereo calibration with Fundamental, Essential, and projection matrix computation
✅ Stereo image rectification and epipolar alignment
✅ Dense disparity map computation using StereoSGBM + WLS filtering
✅ Feature-based matching using AKAZE and epipolar constraint validation
✅ 3D reconstruction via triangulation
✅ Point cloud generation and visualization using Open3D

We have used two aproaches to implement the stereo vision setup:

1. Using Signle Camera by translation of 10cm horizontaly
2. Using Dual Cameras with baseline of 4.5 cm 

Camera Calibration is done using checkboard of (7,9) inner corners with square size of 15mm. For single camera stereo setup, we have used 8 images to clibrate the camera and get reprojection error of 0.2 pixels. Calibration images for the single camera are placed in folder "cal2". Stereo image pairs are placed in the same folder with pictures named as left.jpg and right.jpg which are further used in the pipeline. Run Single_Cam.ipynb file for this approach


For dual camera setup, we have captured 150 images from left and right camera which are stored in the folder named as "left" and "right" folder. Stereo images used in this aproach as placed outside with names "left.png" and "right.png. Both cameras had reprojection error of 0.04 pixels. Run Dual_Camera.ipynb file to see the results obtained fron this approach. Stereo calibration results are stored in the file stereo_calibration.npz file.

Furthermore we have created the GUI for the dual camera stereo system having the following flow: 

1. Select the folder for left and right camera calibration images are placed and calibrate the cameras. Calibration results are shown in the log. 
2. After calibration it will ask you the select the stereo images pairs where the left and right images are given.
3. After selecting the images, it will rectify the image pairs, and it will display the rectified images as a figure and the related results in the log.
4. After this process you can choose to implement the Sparse or Dense approach.
5. By clicking on the sparse button you will get the fundamental matrix, essential matrix, relative rotation matrix and translation matrix (R1,R2 & T). Results are shown in the log. 3D points are reprojected on the original image.
6. Bt clicking on dense button, it will implement the SGBM algorithm on the stereo images and will displayt the depth map and 3d point cloud. 












