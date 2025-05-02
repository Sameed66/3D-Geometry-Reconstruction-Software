# gui.py
import tkinter as tk
from tkinter import filedialog, scrolledtext
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from functions import *

chessboard_size = (7, 9)
square_size = 15

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Calibration and Rectification")

        self.left_folder = ""
        self.right_folder = ""

        self.K_left = None
        self.dist_left = None
        self.K_right = None
        self.dist_right = None

        self.left_image_path = ""
        self.right_image_path = ""

        # Buttons for calibration
        self.select_left_btn = tk.Button(root, text="Select Left Camera Folder", command=self.select_left_folder)
        self.select_left_btn.pack(pady=5)

        self.select_right_btn = tk.Button(root, text="Select Right Camera Folder", command=self.select_right_folder)
        self.select_right_btn.pack(pady=5)

        self.calibrate_btn = tk.Button(root, text="Calibrate", command=self.calibrate)
        self.calibrate_btn.pack(pady=10)

        # Text Box to display results
        self.output_text = scrolledtext.ScrolledText(root, width=80, height=20)
        self.output_text.pack(padx=10, pady=10)

        # Section for rectification
        

    def select_left_folder(self):
        self.left_folder = filedialog.askdirectory(title="Select Left Camera Folder")
        if self.left_folder:
            self.output_text.insert(tk.END, f"Selected Left Folder: {self.left_folder}\n")

    def select_right_folder(self):
        self.right_folder = filedialog.askdirectory(title="Select Right Camera Folder")
        if self.right_folder:
            self.output_text.insert(tk.END, f"Selected Right Folder: {self.right_folder}\n")

    def calibrate(self):
        if not self.left_folder or not self.right_folder:
            self.output_text.insert(tk.END, "Please select both folders before calibrating.\n")
            return

        try:
            # Load images
            left_images = glob.glob(self.left_folder + "/*.png")
            right_images = glob.glob(self.right_folder + "/*.png")

            # Calibrate left camera
            self.output_text.insert(tk.END, "Calibrating Left Camera...\n")
            self.K_left, self.dist_left, error_left = calibrate_single_camera(left_images, chessboard_size, square_size)
            self.output_text.insert(tk.END, f"Left Camera Matrix:\n{self.K_left}\n")
            self.output_text.insert(tk.END, f"Left Distortion Coefficients:\n{self.dist_left.ravel()}\n")
            self.output_text.insert(tk.END, f"Left Reprojection Error: {error_left}\n\n")

            # Calibrate right camera
            self.output_text.insert(tk.END, "Calibrating Right Camera...\n")
            self.K_right, self.dist_right, error_right = calibrate_single_camera(right_images, chessboard_size, square_size)
            self.output_text.insert(tk.END, f"Right Camera Matrix:\n{self.K_right}\n")
            self.output_text.insert(tk.END, f"Right Distortion Coefficients:\n{self.dist_right.ravel()}\n")
            self.output_text.insert(tk.END, f"Right Reprojection Error: {error_right}\n\n")

            self.output_text.insert(tk.END, "Calibration Complete.\n")
            self.select_left_image_btn = tk.Button(root, text="Select Left Image", command=self.select_left_image)
            self.select_left_image_btn.pack(pady=5)
            self.select_right_image_btn = tk.Button(root, text="Select Right Image", command=self.select_right_image)
            self.select_right_image_btn.pack(pady=5)
            self.rectify_btn = tk.Button(root, text="Rectify Images", command=self.rectify)
            self.rectify_btn.pack(pady=10)
        except Exception as e:
            self.output_text.insert(tk.END, f"Error during calibration: {e}\n")

    def select_left_image(self):
        self.left_image_path = filedialog.askopenfilename(title="Select Left Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if self.left_image_path:
            self.output_text.insert(tk.END, f"Selected Left Image: {self.left_image_path}\n")

    def select_right_image(self):
        self.right_image_path = filedialog.askopenfilename(title="Select Right Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if self.right_image_path:
            self.output_text.insert(tk.END, f"Selected Right Image: {self.right_image_path}\n")
    
        
    def rectify(self):
        if not self.left_image_path or not self.right_image_path:
            self.output_text.insert(tk.END, "Please select both images before rectifying.\n")
            return

        if self.K_left is None or self.K_right is None:
            self.output_text.insert(tk.END, "Please calibrate the cameras first.\n")
            return

        try:
            # Load images
            self.img1 = cv2.imread(self.left_image_path)
            img2 = cv2.imread(self.right_image_path)

            # Define transformation matrix T
            T = np.eye(4)
            T[:3, 3] = np.array([0.45, 0, 0])  # Example translation 4.5 cm along x-axis

            # Rectify images
            self.rectified_img1, self.rectified_img2, self.Q = rectify_images(self.img1, img2, self.K_left, self.dist_left, self.K_right, self.dist_right, T)
            cv2.imwrite("rectified_img1.png",self.rectified_img1)
            cv2.imwrite("rectified_img2.png",self.rectified_img2)
            def draw_epipolar_lines(image, color=(0, 255, 0), interval=20):
                """Draw horizontal epipolar lines at fixed vertical intervals."""
                img_with_lines = image.copy()
                h, w = image.shape[:2]
                for y in range(0, h, interval):
                    cv2.line(img_with_lines, (0, y), (w, y), color, 1)
                return img_with_lines

            # Draw epipolar lines
            epipolar_img1 = draw_epipolar_lines(self.rectified_img1, color=(0, 255, 0))
            epipolar_img2 = draw_epipolar_lines(self.rectified_img2, color=(255, 0, 0))

            # Show side-by-side for comparison
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(epipolar_img1, cv2.COLOR_BGR2RGB))
            plt.title("Rectified lefttt with Epipolar Lines")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(epipolar_img2, cv2.COLOR_BGR2RGB))
            plt.title("Rectified Right with Epipolar Lines")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            # # Show images using matplotlib
            # plt.figure()
            # plt.imshow(cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2RGB))
            # plt.title("Rectified Left Image")
            # plt.axis('off')
            # plt.show()

            # plt.figure()
            # plt.imshow(cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2RGB))
            # plt.title("Rectified Right Image")
            # plt.axis('off')
            # plt.show()

            # Print Q matrix
            self.output_text.insert(tk.END, f"Q Matrix:\n{self.Q}\n")

            self.select_sparse = tk.Button(root, text="Sparse", command=self.select_sparse)
            self.select_sparse.pack(pady=5)
            self.select_denses = tk.Button(root, text="Dense", command=self.select_dense)
            self.select_denses.pack(pady=5)

        except Exception as e:
            self.output_text.insert(tk.END, f"Error during rectification: {e}\n")
    def select_dense(self):
        dense_3d_reconstruction(self.Q)
    def select_sparse(self):
        F, E, R1, R2, t, det_R1, det_R2,det_E_string=sparse(self.rectified_img1,self.rectified_img2,self.K_left,self.img1)

        self.output_text.insert(tk.END, f"Fundamental Matrix:\n{F}\nEssential  Matrix:\n{E}\n{det_E_string}\nRotation Matrix R1:\n{R1}\nRotation Matrix R2:\n{R2}\nTranslation Matrix:\n{t}\nDeterminant of R1:\n{det_R1}\nDeterminant of R2:\n{det_R2}\n")
        

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()
