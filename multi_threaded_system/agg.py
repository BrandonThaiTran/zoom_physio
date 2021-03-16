import imutils
from imutils.video import VideoStream, FileVideoStream
from imutils import face_utils
import cv2
from time import time, sleep, perf_counter, process_time
import numpy as np
import dlib
from collections import OrderedDict

from utils import *

import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, lfilter, find_peaks
import pandas as pd

# Face streamer class
class face_streamer:
    def __init__(self, predictor_path, filename = None):
        self.filename = filename

        # Initialize dlib's face detector (HOG-based)
        self.detector = dlib.get_frontal_face_detector()

        # Create landmark predictor.
        self.predictor = dlib.shape_predictor(predictor_path)

        self.facial_landmarks_idxs = OrderedDict([
            ("face", (0, 26)),
            ("left_eye", (37, 42)),
            ("right_eye", (43, 48)),
        ])

        # Colors to choose from
        self.colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                       (168, 100, 168), (158, 163, 32),
                       (163, 38, 32), (180, 42, 220), (100, 150, 250)]

        # Define the width
        self.width = 400

        # RGB signals
        self.red_window = []
        self.green_window = []
        self.blue_window = []
        self.red = []
        self.green = []
        self.blue = []

        # RPY signals
        self.roll_window = []
        self.pitch_window = []
        self.yaw_window = []
        self.roll = []
        self.pitch = []
        self.yaw = []

        # POS Signals
        self.S1 = []
        self.S2 = []
        self.P = []

        # rPPG
        self.rppg = []

        # fft signals
        self.roll_fft = []
        self.pitch_fft = []
        self.yaw_fft = []

        self.rppg_fft = []

        # RMNS
        self.combined_rpy_fft = []
        self.rppg_fft_rmns = []

        # Freq filtered signals
        self.rppg_filtered = []

        # After zmean
        self.rppg_zmean = [0]

        self.frame_count = 0
        self.num_shifts = 0

    def set_display(self, display_face_bb = False, display_landmarks = False, display_overlay = False,
                    display_aam = False, display_pose_unstable = False, display_pose_stable = False,
                    display_pose_axis = False):
        # Update display parameters
        self.display_face_bb = display_face_bb
        self.display_landmarks = display_landmarks
        self.display_overlay = display_overlay
        self.display_aam = display_aam
        self.display_pose_unstable = display_pose_unstable
        self.display_pose_stable = display_pose_stable
        self.display_pose_axis = display_pose_axis

    def stream(self, display_face_bb = False, display_landmarks = False, display_overlay = False,
               display_aam = False, display_pose_unstable = False, display_pose_stable = False,
               display_pose_axis = False):

        # Start the stream
        self.start_stream()

        self.window_start_time = perf_counter()

        # Loop and stream
        while True:
            # Process each frame
            self.process_frame()

            # Show the frame
            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1) & 0xFF

            # If the `q` key was pressed, break from the loop.
            if key == ord("q") or self.vs.stopped:
                print('break')
                break

        print('Final process')
        # Process the signals
        self.process_signals()
        # Do some cleanup
        self.end_stream()

    def start_stream(self):
        print("[INFO] camera sensor warming up...")
        self.set_params()
        sleep(1.0)

    def end_stream(self):
        # Do some cleanup
        cv2.destroyAllWindows()
        self.vs.stop()
        del self.vs
        self.num_frames = len(self.red)
        self.frame_vector = range(self.num_frames)
        self.rppg_frame_vector = range(len(self.rppg_zmean))
        self.calculate_output()

    def set_params(self):
        # Start default camera
        if self.filename:
            self.vs = FileVideoStream(self.filename).start()
        else:
            self.vs = VideoStream(src=0).start()

        # Number of frames to capture
        num_frames = 10

        print("Capturing {0} frames".format(num_frames))

        # Start time
        start = time()

        # Grab a few frames
        for i in range(0, num_frames):
            frame = self.vs.read()
            frame = imutils.resize(frame, width = self.width)

        # End time
        end = time()

        # Time elapsed
        seconds = end - start
        print (f"Time taken to capture {num_frames}: {seconds} seconds")

        # Calculate frames per second
        # Round to 30 or 60
        fps  = num_frames / seconds
        possible_rates = [30, 60] # these are the possible rates that we can round to
        self.fps = possible_rates[min(range(len(possible_rates)), key = lambda i: abs(possible_rates[i]-fps))]
        #         self.fps = 30
        print("Estimated frames per second : {0}".format(self.fps))

        # Find image dimensions and make the stablizer
        frame = imutils.resize(self.vs.read(), width=self.width)
        self.height, _, _ = frame.shape
        self.pose_estimator = PoseEstimator(img_size=(self.height, self.width))
        self.pose_stabilizers = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1)
                                 for _ in range(6)]

        # Set the frame limit
        if self.fps == 30:
            self.frame_limit = 256
        else:
            self.frame_limit = 512

        # Time limit
        self.time_limit = self.frame_limit/self.fps
        print(f'Time limit: {self.time_limit:2.2} seconds \nFrame limit: {self.frame_limit}')

        self.frame_limit_vector = range(int(self.frame_limit))

    # Processes the signals when the frame limit has been reached
    def process_signals(self):
        self.resample_signals()
        self.apply_pos()
        self.apply_signal_filtering()
        self.apply_post_processing()
        self.append_window_signals()

    # Processes the frame by updating numerical values and drawing on it (if specified)
    def process_frame(self):
        # Process and append the signals if the frame_limit is reached
        if self.frame_count >= self.frame_limit:
            self.process_signals()
        # Process and append the signals if the time limit is reached
        #         if perf_counter() - self.window_start_time >= self.time_limit:
        #             self.process_signals()

        self.find_face()
        self.loop_faces()
        self.frame_count += 1

    # Finds faces in the image
    def find_face(self):
        # Read and resize the frame
        self.frame = self.vs.read()
        self.frame = imutils.resize(self.vs.read(), width=self.width)
        # Get grayscale image and extract the bounding boxes with the detector
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.rects = self.detector(self.gray, 0)

    # Loops over the faces to update values and to draw on the frame
    def loop_faces(self):
        for rect in self.rects:
            # Get the bounding box
            (self.bX, self.bY, self.bW, self.bH) = face_utils.rect_to_bb(rect)
            # Determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array.
            self.shape = face_utils.shape_to_np(self.predictor(self.gray, rect))
            # Set the facial points for the roi
            self.set_face_points()
            # Apply the aam
            self.apply_aam()
            # Get RGB values
            self.update_rgb()
            # Try pose estimation
            self.pose = self.pose_estimator.solve_pose_by_68_points(
                self.shape.astype('float'))
            # Stabilize the pose
            self.stablize_pose()
            # Update RPY values
            self.update_rpy()
            # If a face is found, draw on the face
            if self.shape.any():
                self.draw_frame()

    def update_rgb(self):
        self.red_window.append(np.sum(self.aam[:,:,0])/ self.num_aam_pixels)
        self.green_window.append(np.sum(self.aam[:,:,1])/ self.num_aam_pixels)
        self.blue_window.append(np.sum(self.aam[:,:,2])/ self.num_aam_pixels)

    def update_rpy(self):
        self.roll_window.append(self.steady_pose[0][2] * -1)
        self.pitch_window.append(self.steady_pose[0][1])
        self.yaw_window.append(self.steady_pose[0][0])

    # Applies the AAM by turning all cells outside of the face to 0
    def apply_aam(self):
        self.aam = np.zeros_like(self.frame)

        # Initialize masks
        feature_mask = np.zeros((self.frame.shape[0], self.frame.shape[1]))
        l_eye_mask = np.zeros((self.frame.shape[0], self.frame.shape[1]))
        r_eye_mask = np.zeros((self.frame.shape[0], self.frame.shape[1]))
        custom_mask = np.zeros((self.frame.shape[0], self.frame.shape[1]))

        # Define hulls for the facial components
        hull = cv2.convexHull(self.face_points)
        hull_left_eye = cv2.convexHull(self.left_eye_points)
        hull_right_eye = cv2.convexHull(self.right_eye_points)
        custom_hull = cv2.convexHull(self.custom_points)

        # Fill the convex hulls for 1s, which mean that that pixel location is in the ROI.
        feature_mask = cv2.fillConvexPoly(feature_mask, hull, 1).astype(np.bool)
        l_eye_mask = cv2.fillConvexPoly(l_eye_mask, hull_left_eye, 1).astype(np.bool)
        r_eye_mask = cv2.fillConvexPoly(r_eye_mask, hull_right_eye, 1).astype(np.bool)
        custom_mask = cv2.fillConvexPoly(custom_mask, custom_hull, 1).astype(np.bool)

        # Use XOR to make a boolean mask of pixels inside our ROI
        final_mask = np.logical_xor(feature_mask, l_eye_mask)
        final_mask = np.logical_xor(final_mask, r_eye_mask)

        custom_final_mask = np.logical_xor(custom_mask, l_eye_mask)
        self.custom_final_mask = np.logical_xor(custom_final_mask, r_eye_mask)

        # Final AAM
        self.aam[self.custom_final_mask] = self.frame[self.custom_final_mask]
        self.num_aam_pixels = np.sum(self.custom_final_mask)

        # Apply POS to combine rgb signal in to rPPG signal
    def apply_pos(self):
        self.red_window = self.normalize_signal(self.red_window)
        self.green_window = self.normalize_signal(self.green_window)
        self.blue_window = self.normalize_signal(self.blue_window)
        C = np.array([self.red_window, self.green_window, self.blue_window])
        mean_color = np.mean(C, axis=1)
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        Cn = np.matmul(diag_mean_color_inv,C)
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        self.S_window = np.matmul(projection_matrix,Cn)
        std = np.array([1,np.std(self.S_window[0,:])/np.std(self.S_window[1,:])])
        self.P_window = np.matmul(std,self.S_window)
        self.rppg_window = self.P_window-np.mean(self.P_window)

    def apply_signal_filtering(self):
        self.apply_rmns()
        self.apply_wnb_filter()

    # empircally play with the normalizing in the freq domain
    # normalize RPY and RGB
    # RPY -45 and +45 -> []
    # RPY to freq -> normalize -> combine (average)
    # todo - fix scale of rpy_fft signal
    # Applies Rhythmic Motion Noise Suppresion
    def apply_rmns(self):
        # Method 1: 1) temporal normalize RPY 2) FFT of normalized signals 3) combine
        # Normalize RPY signals
        self.roll_window = self.normalize_signal(self.roll_window)
        self.pitch_window = self.normalize_signal(self.pitch_window)
        self.yaw_window = self.normalize_signal(self.yaw_window)
        # Find fft of RPY and rPPG signals
        self.rppg_fft_window = np.abs(fft(self.rppg_window))
        self.roll_fft_window = np.abs(fft(self.roll_window))
        self.pitch_fft_window = np.abs(fft(self.pitch_window))
        self.yaw_fft_window = np.abs(fft(self.yaw_window))
        # Combine rpy_fft signals via averaging (divide by 3)
        self.combined_rpy_fft_window = (self.roll_fft_window + self.pitch_fft_window + self.yaw_fft_window)/3
        # Normalization
        #         self.combined_rpy_fft_window = self.combined_rpy_fft_window/np.mean(self.combined_rpy_fft_window)

        #         # Method 2: 1) FFT 2) normalize FFT 3) combine
        #         # take FFT
        #         self.rppg_fft_window = np.abs(fft(self.rppg_window))
        #         self.roll_fft_window = np.abs(fft(self.roll_window))
        #         self.pitch_fft_window = np.abs(fft(self.pitch_window))
        #         self.yaw_fft_window = np.abs(fft(self.yaw_window))
        #         # Normalize
        #         self.roll_fft_window = self.normalize_signal(self.roll_fft_window)
        #         self.pitch_fft_window = self.normalize_signal(self.pitch_fft_window)
        #         self.yaw_fft_window = self.normalize_signal(self.yaw_fft_window)
        #         # Combine rpy_fft signals via averaging (divide by 3)
        #         self.combined_rpy_fft_window = (self.roll_fft_window + self.pitch_fft_window + self.yaw_fft_window)/3

        #         # Method 3: 1) FFT 2) combine 3)normalize
        #         # take FFT
        #         self.rppg_fft_window = np.abs(fft(self.rppg_window))
        #         self.roll_fft_window = np.abs(fft(self.roll_window))
        #         self.pitch_fft_window = np.abs(fft(self.pitch_window))
        #         self.yaw_fft_window = np.abs(fft(self.yaw_window))
        #         # Combine rpy_fft signals via averaging (divide by 3)
        #         self.combined_rpy_fft_window = self.normalize_signal((self.roll_fft_window + self.pitch_fft_window + self.yaw_fft_window)/3)

        # Apply RMNS
        #         self.rppg_fft_rmns_window = self.rppg_fft_window - self.combined_rpy_fft_window
        # Without RMNS
        self.rppg_fft_rmns_window = self.rppg_fft_window

    # TODO - Tune
    def apply_wnb_filter(self):
        bandwidth = .2
        nyq = 0.5 * self.fps
        # Find max freq
        max_freq = self.find_highest_freq(self.rppg_fft_rmns_window)
        # Make band
        freq_band = [(max_freq + i*bandwidth/2)/nyq for i in [-1, 1]]
        #         print(freq_band)
        # Butterworth filter
        N = 5 # butterworth signal order
        b, a = butter(N, freq_band, btype='bandpass')
        # use bandpass filter
        self.rppg_filtered_window = lfilter(b, a, self.rppg_fft_rmns_window)
    #         self.rppg_freq_filtered_window = np.abs(ifft(self.rppg_fft_rmns_window))

    # Finds highest frequency in hz
    def find_highest_freq(self, signal):
        freq_hz = abs(fftfreq(signal.shape[0]) * self.fps)
        max_idx = np.argmax(signal)
        max_freq = freq_hz[max_idx]
        return max_freq

    def apply_post_processing(self):
        self.rppg_zmean_window = (self.rppg_filtered_window - np.mean(self.rppg_filtered_window))/np.std(self.rppg_filtered_window)
        self.rppg_len = len(self.rppg_zmean)
        # determine number of frames to shift
        seg_t = self.time_limit / 2
        l = 1 # num frames to shift
        #         l = int(self.fps * seg_t)
        #         self.num_shifts += 1
        #         print('n shift', self.num_shifts)
        #         self.rppg_zmean_window[:l] = [rppg + self.rppg_zmean[self.rppg_len-l+i] for (i,rppg) in enumerate(self.rppg_zmean_window[:l])]
        #         self.rppg_zmean = [rppg/self.num_shifts for rppg in self.rppg_zmean]
        self.rppg_zmean = self.rppg_zmean[:-1*l]
        self.rppg_zmean.extend(self.rppg_zmean_window)
        self.rppg_len = len(self.rppg_zmean)
    #         H[t:t+l] = H[t:t+l] +  (P-np.mean(P))

    def stablize_pose(self):
        self.steady_pose = []
        pose_np = np.array(self.pose).flatten()
        for value, ps_stb in zip(pose_np, self.pose_stabilizers):
            ps_stb.update([value])
            self.steady_pose.append(ps_stb.state[0])
        self.steady_pose = np.reshape(self.steady_pose, (-1, 3))

    def set_face_points(self):
        # x,y locations of the facial landmarks
        self.face_points = self.shape[self.facial_landmarks_idxs['face'][0]:self.facial_landmarks_idxs['face'][1]]
        self.left_eye_points = self.shape[self.facial_landmarks_idxs['left_eye'][0]:self.facial_landmarks_idxs['left_eye'][1]]
        self.right_eye_points = self.shape[self.facial_landmarks_idxs['right_eye'][0]:self.facial_landmarks_idxs['right_eye'][1]]

        # Define custom ROI
        custom_roi = [0,1,2,3,13,14,15,16,17,18,19,20,21,22,23,24,25,26,33]
        avg_1 = np.asarray([np.mean([self.shape[3][0], self.shape[48][0]],dtype=np.int64), np.mean([self.shape[3][1],self.shape[48][1]],dtype=np.int64)])
        avg_2 = np.asarray([np.mean([self.shape[13][0], self.shape[54][0]],dtype=np.int64), np.mean([self.shape[13][1],self.shape[54][1]])],dtype=np.int64)
        points = [self.shape[i] for i in custom_roi]
        points.extend([avg_1, avg_2])
        self.custom_points = np.asarray(points)

    # normalize RGB signal by dividing by mean
    def normalize_signal(self, signal):
        signal = signal/np.mean(signal)
        return signal

    def resample_signals(self):
        frame_collected_vector = range(len(self.red_window))
        # RGB
        self.red_window = np.interp(self.frame_limit_vector, frame_collected_vector,self.red_window)
        self.green_window = np.interp(self.frame_limit_vector, frame_collected_vector,self.green_window)
        self.blue_window = np.interp(self.frame_limit_vector, frame_collected_vector,self.blue_window)

        # RPY
        self.roll_window = np.interp(self.frame_limit_vector, frame_collected_vector,self.roll_window)
        self.pitch_window = np.interp(self.frame_limit_vector, frame_collected_vector,self.pitch_window)
        self.yaw_window = np.interp(self.frame_limit_vector, frame_collected_vector,self.yaw_window)

    def append_window_signals(self):
        print('append called')
        # RGB signals
        self.red.extend(self.red_window)
        self.green.extend(self.green_window)
        self.blue.extend(self.blue_window)

        # RPY signals
        self.roll.extend(self.roll_window)
        self.pitch.extend(self.pitch_window)
        self.yaw.extend(self.yaw_window)

        # POS Signals
        self.S1.extend(self.S_window[0,:])
        self.S2.extend(self.S_window[1,:])
        self.P.extend(self.P_window)

        # rPPG
        self.rppg.extend(self.rppg_window)

        # fft signals
        self.roll_fft.extend(self.roll_fft_window)
        self.pitch_fft.extend(self.pitch_fft_window)
        self.yaw_fft.extend(self.yaw_fft_window)

        self.rppg_fft.extend(self.rppg_fft_window)

        # RMNS
        self.combined_rpy_fft.extend(self.combined_rpy_fft_window)
        self.rppg_fft_rmns.extend(self.rppg_fft_rmns_window)

        # Freq filtered signals
        self.rppg_filtered.extend(self.rppg_filtered_window)

        # Zero mean rPPG
        #         self.rppg_zmean.extend(self.rppg_zmean_window)

        self.reset_window_signals()

    def reset_window_signals(self):
        # Frame count
        #         print('frame count: ', self.frame_count)
        self.frame_count = 0
        # Time
        #         print('window time: ', perf_counter() - self.window_start_time)
        self.window_start_time = perf_counter()
        # RGB signals
        self.red_window = []
        self.green_window = []
        self.blue_window = []

        # RPY signals
        self.roll_window = []
        self.pitch_window = []
        self.yaw_window = []

    def calculate_output(self):
        # Find the peaks
        detected_peak_idxs = find_peaks(self.rppg_zmean, distance = self.fps/2)[0]
        self.time_vector = [frame * (1/self.fps) for frame in range(self.rppg_len)] # in seconds
        self.peaks = [self.time_vector[detected_peak_idx] for detected_peak_idx in detected_peak_idxs]
        # IBIs
        self.IBIs = self.find_IBIs(self.peaks)
        # hr and HRV
        self.hr = self.find_hr(self.IBIs)
        self.rmssd, self.sdnn = self.find_hrv(self.IBIs)

    # Finds IBIs in seconds
    def find_IBIs(self, peaks):
        IBIs = []
        for i in range(len(peaks)-1):
            IBIs.append(peaks[i+1] - peaks[i])
        return IBIs

    def find_hr(self, IBIs):
        IBI_mean = np.average(IBIs)
        hr = 1/IBI_mean * 60
        return hr

    # TODO - multiply by 1000, not 100
    def find_hrv(self, IBIs):
        rmssd = self.find_rmssd(IBIs) * 100
        sdnn = self.find_sdnn(IBIs) * 100
        return rmssd, sdnn

    def find_rmssd(self, IBIs):
        N = len(IBIs)
        ssd = 0
        for i in range(N-1):
            ssd += (IBIs[i+1] - IBIs[i])**2
        rmssd = np.sqrt(ssd/(N-1))
        return rmssd

    def find_sdnn(self, IBIs):
        sdnn = np.std(IBIs)
        return sdnn

    def draw_frame(self):
        if self.display_aam:
            self.draw_overlay()
        # Display bounding box if true
        if self.display_face_bb:
            self.draw_face_bb()
        # Display facial landmarks if true
        if self.display_landmarks:
            self.draw_landmarks()
        # Display the landmark overlay if true
        if self.display_overlay:
            self.draw_overlay()
        # Display the pose if true
        if self.display_pose_unstable or self.display_pose_stable:
            self.draw_pose()
        # Display the pose axis if true
        if self.display_pose_axis:
            self.draw_pose_axis()

    def draw_face_bb(self):
        # Draw the bounding box on the frame
        cv2.rectangle(self.frame, (self.bX, self.bY),
                      (self.bW+self.bX, self.bH+self.bY), (0, 255, 0), 1)

    def draw_landmarks(self):
        for (name, (i, j)) in self.facial_landmarks_idxs.items():
            # Loop over the subset of facial landmarks, drawing the specific face part
            for (x, y) in self.shape[i:j]:
                cv2.circle(self.frame, (x, y), 1, (0, 0, 255), -1)

    # Displays the overlay of the landmarks
    def draw_overlay(self, alpha=0.75):
        # Apply the transparent overlay
        #         cv2.addWeighted(self.aam, alpha, self.frame, 1 - alpha, 0,self.frame)
        # apply a color
        self.frame[self.custom_final_mask] = self.colors[0]

    def draw_pose(self):
        # Display the initial pose annotation if true
        if self.display_pose_unstable:
            self.pose_estimator.draw_annotation_box(
                self.frame, self.pose[0], self.pose[1],
                color=(255, 128, 128))
        # Display the stablized pose annotation if true
        if self.display_pose_stable:
            self.pose_estimator.draw_annotation_box(
                self.frame, self.steady_pose[0], self.steady_pose[1],
                color=(128, 255, 128))

    def draw_pose_axis(self):
        self.pose_estimator.draw_axis(self.frame,
                                      self.steady_pose[0], self.steady_pose[1])

    def plot_rgb(self):
        plt.title('Normalized RGB values as a function of frames')
        plt.plot(self.frame_vector, self.red, color='red', label = 'Red')
        plt.plot(self.frame_vector, self.green, color='green', label='Green')
        plt.plot(self.frame_vector, self.blue, color='blue', label='Blue')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend(loc = 'upper right')

    def plot_projected_signal(self):
        plt.title('Projected Signals')
        plt.plot(self.frame_vector, self.S1)
        plt.plot(self.frame_vector, self.S2)
        plt.xlabel('Frame')
        plt.ylabel('Value')

    def plot_rppg(self):
        plt.title('rPPG signal')
        plt.plot(self.frame_vector, self.rppg)
        plt.xlabel('Frame')
        plt.ylabel('Value')

    def plot_rpy(self):
        plt.title('RPY values as a function of frames')
        plt.plot(self.frame_vector, self.roll, color='cyan', label = 'Roll')
        plt.plot(self.frame_vector, self.pitch, color='magenta', label='Pitch')
        plt.plot(self.frame_vector, self.yaw, color='yellow', label='Yaw')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend(loc = 'upper right')

    def plot_fft(self):
        plt.subplot(1,2,1)
        plt.title('FFT of RPY signals')
        plt.plot(self.frame_vector, self.roll_fft, color='cyan', label = 'Roll')
        plt.plot(self.frame_vector, self.pitch_fft, color='magenta', label='Pitch')
        plt.plot(self.frame_vector, self.yaw_fft, color='yellow', label='Yaw')
        plt.legend(loc = 'upper right')
        plt.subplot(1,2,2)
        plt.title('FFT rPPG signals')
        plt.plot(self.frame_vector, self.rppg_fft, label = 'rPPG')

    def plot_combined_rpy_fft(self):
        plt.subplot(1,2,1)
        plt.title('Combined FFT of RPY signal')
        plt.plot(self.frame_vector, self.combined_rpy_fft)
        plt.subplot(1,2,2)
        plt.title('Combined FFT of rPPG signal')
        plt.plot(self.frame_vector, self.rppg_fft)

    def plot_rppg_rmns(self):
        plt.title('FFT of rPPG signal after Rhythmic Noise Suppression')
        # freq domain
        plt.plot(self.frame_vector, self.rppg_fft_rmns)

    def plot_rppg_filtered(self):
        plt.title('rPPG after Frequency Filter')
        plt.plot(self.frame_vector, self.rppg_filtered)

    def plot_rppg_zmean(self):
        plt.title('rPPG with zero mean')
        plt.plot(self.rppg_frame_vector,self.rppg_zmean)

    def plot_peaks(self):
        plt.plot(self.time_vector, self.rppg_zmean)
        plt.xlabel('Time (ms)')
        for peak in self.peaks:
            plt.axvline(x = peak, color = 'r')
        print(f'HR (BPM): {self.hr}')
        print(f'HRV RMSSD (ms) {self.rmssd} SDNN: {self.sdnn}')
        IBI_min = np.min(self.IBIs)
        IBI_max = np.max(self.IBIs)
        print(f'IBI range: {IBI_max - IBI_min} Low: {IBI_min} High: {IBI_max}')

predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
# filename = '/media/brandon/Seagate HDD/datasets/vicarPPG/Videos/01-base.mp4' # full vid
# filename = '/home/brandon/Downloads/test.mp4' # shorter vid
filename = None
fs = face_streamer(predictor_path, filename = filename)
# If display_aam is true, this is all you will see
fs.set_display(display_face_bb = False, display_landmarks = True, display_overlay = False, display_aam = True,
               display_pose_unstable = False, display_pose_stable = True, display_pose_axis = True)
fs.stream()

# Graph RGB
fs.plot_rgb()

fs.plot_rppg_zmean()