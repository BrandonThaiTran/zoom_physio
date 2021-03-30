import imutils
from imutils.video import VideoStream, FileVideoStream
from imutils import face_utils
import cv2
from time import time, sleep, perf_counter, process_time
import numpy as np
import dlib
from collections import OrderedDict
from utils import *
import queue
import threading
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from multiprocessing import Process, Manager
from dash.dependencies import Input, Output


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

frame_info = queue.Queue()
red = queue.Queue()
green = queue.Queue()
blue = queue.Queue()
roll = queue.Queue()
pitch = queue.Queue()
yaw = queue.Queue()
S1 = queue.Queue()
S2 = queue.Queue()
P = queue.Queue()
rppg = queue.Queue()
roll_fft = queue.Queue()
pitch_fft = queue.Queue()
yaw_fft = queue.Queue()
rppg_fft = queue.Queue()
combined_rpy_fft = queue.Queue()
rppg_fft_rmns = queue.Queue()
rppg_filtered = queue.Queue()
rppg_zmean = queue.Queue()


class FaceStreamer:

    def __init__(self, predictor_path, filename=None):

        self.filename = filename

        # Initialize dlib's face detector (HOG-based)
        self.detector = dlib.get_frontal_face_detector()

        # Create landmark predictor.
        self.predictor = dlib.shape_predictor(predictor_path)

        self.facial_landmarks_ids = OrderedDict([
            ("face", (0, 26)),
            ("left_eye", (37, 42)),
            ("right_eye", (43, 48)),
        ])

        self.width = 400
        self.frame_count = 0

        self.colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                       (168, 100, 168), (158, 163, 32),
                       (163, 38, 32), (180, 42, 220), (100, 150, 250)]

        self.frame = None

        # for overlap add
        self.num_shifts = 0

    def stream(self):

        self._start_stream()

        self.window_start_time = perf_counter()

        while True:
            self.frame = imutils.resize(self.vs.read(), width=self.width)
            self._process_frame()
            self.frame_count += 1

            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1) & 0xFF

            # If the `q` key was pressed, break from the loop.
            if key == ord("q") or self.vs.stopped:
                print('break')
                break

        self._end_stream()

    def _start_stream(self):
        print("[INFO] camera sensor warming up...")
        if self.filename:
            self.vs = FileVideoStream(self.filename).start()
        else:
            self.vs = VideoStream(src=0).start()

        self._set_params()
        sleep(1.0)

    def _end_stream(self):
        cv2.destroyAllWindows()
        self.vs.stop()
        del self.vs

    def _set_params(self):

        # Find image dimensions and make the stablizer
        frame = imutils.resize(self.vs.read(), width=self.width)
        self.height, _, _ = frame.shape
        self.pose_estimator = PoseEstimator(img_size=(self.height, self.width))
        self.pose_stabilizers = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1)
                                 for _ in range(6)]

    def _process_frame(self):
        self._find_faces()
        self._loop_faces()

    def _find_faces(self):
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.rects = self.detector(self.gray, 0)

    def _loop_faces(self):

        if len(self.rects) > 1:
            raise ValueError("Too many faces")

        for rect in self.rects:
            # Get the bounding box
            (self.bX, self.bY, self.bW, self.bH) = face_utils.rect_to_bb(rect)
            # Determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array.
            self.shape = face_utils.shape_to_np(self.predictor(self.gray, rect))
            # Set the facial points for the roi
            self._find_face_points()
            # Apply the aam
            self._apply_aam()
            # Get RGB values
            rgb_triple = self._update_rgb()
            # Try pose estimation
            self.pose = self.pose_estimator.solve_pose_by_68_points(
                self.shape.astype('float'))
            # Stabilize the pose
            self._stablize_pose()
            # Update RPY values
            rpy_triple = self._update_rpy()

            # If a face is found, draw on the face
            if self.shape.any():
                self._draw_frame()
            frame_info.put((perf_counter(), self.frame_count, rgb_triple, rpy_triple))

    def _find_face_points(self):

        # Define custom ROI
        custom_roi = [0,1,2,3,13,14,15,16,17,18,19,20,21,22,23,24,25,26,33]
        avg_1 = np.asarray([np.mean([self.shape[3][0], self.shape[48][0]],dtype=np.int64), np.mean([self.shape[3][1],self.shape[48][1]],dtype=np.int64)])
        avg_2 = np.asarray([np.mean([self.shape[13][0], self.shape[54][0]],dtype=np.int64), np.mean([self.shape[13][1],self.shape[54][1]])],dtype=np.int64)
        points = [self.shape[i] for i in custom_roi]
        points.extend([avg_1, avg_2])

        self.face_points = self.shape[self.facial_landmarks_ids['face'][0]:self.facial_landmarks_ids['face'][1]]
        self.left_eye_points = self.shape[self.facial_landmarks_ids['left_eye'][0]:self.facial_landmarks_ids['left_eye'][1]]
        self.right_eye_points = self.shape[self.facial_landmarks_ids['right_eye'][0]:self.facial_landmarks_ids['right_eye'][1]]
        self.custom_points = np.asarray(points)

    def _apply_aam(self):
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
        feature_minus_left = np.logical_xor(feature_mask, l_eye_mask)
        feature_minus_eyes = np.logical_xor(feature_minus_left, r_eye_mask)

        self.custom_final_mask = np.logical_xor(custom_mask, l_eye_mask)
        self.custom_final_mask = np.logical_xor(self.custom_final_mask, r_eye_mask)

        # Final AAM
        self.aam[self.custom_final_mask] = self.frame[self.custom_final_mask]
        self.num_aam_pixels = np.sum(self.custom_final_mask)

    def _update_rgb(self):
        red_ = np.sum(self.aam[:,:,0])/ self.num_aam_pixels
        green_ = np.sum(self.aam[:,:,1])/ self.num_aam_pixels
        blue_ = np.sum(self.aam[:,:,2])/ self.num_aam_pixels

        return red_, green_, blue_

    def _stablize_pose(self):
        self.steady_pose = []
        pose_np = np.array(self.pose).flatten()
        for value, ps_stb in zip(pose_np, self.pose_stabilizers):
            ps_stb.update([value])
            self.steady_pose.append(ps_stb.state[0])
        self.steady_pose = np.reshape(self.steady_pose, (-1, 3))

    def _update_rpy(self):
        roll_ = self.steady_pose[0][2] * -1
        pitch_ = self.steady_pose[0][1]
        yaw_ = self.steady_pose[0][0]

        return roll_, pitch_, yaw_

    def _draw_frame(self):

        self._draw_overlay()
        self._draw_face_bb()
        self._draw_landmarks()
        self._draw_pose_axis()

    def _draw_face_bb(self):
        # Draw the bounding box on the frame
        cv2.rectangle(self.frame, (self.bX, self.bY),
                      (self.bW+self.bX, self.bH+self.bY), (0, 255, 0), 1)

    def _draw_landmarks(self):
        for (name, (i, j)) in self.facial_landmarks_ids.items():
            # Loop over the subset of facial landmarks, drawing the specific face part
            for (x, y) in self.shape[i:j]:
                cv2.circle(self.frame, (x, y), 1, (0, 0, 255), -1)

    # Displays the overlay of the landmarks
    def _draw_overlay(self, alpha=0.75):
        # Apply the transparent overlay
        #         cv2.addWeighted(self.aam, alpha, self.frame, 1 - alpha, 0,self.frame)
        # apply a color
        self.frame[self.custom_final_mask] = self.colors[0]

    def _draw_pose_axis(self):
        self.pose_estimator.draw_axis(self.frame,
                                      self.steady_pose[0], self.steady_pose[1])


class SignalProcessor:

    def __init__(self):
        self.frames_in_window = 128
        self.fps = 30
        self.red_window, self.green_window, self.blue_window = None, None, None
        self.roll_window, self.pitch_window, self.yaw_window = None, None, None
        self.frame_timestamps_window, self.frame_ids_window, self.rgb_values_window, self.rpy_values_window = None, None, None, None
        self.S_window, self.P_window, self.rppg_window = None, None, None
        self.rppg_fft_window, self.roll_fft_window, self.yaw_fft_window, self.pitch_fft_window =  None, None, None, None
        self.combined_rpy_fft_window, self.rppg_fft_rmns_window = None, None
        self.rppg_filtered_window = None
        self.rppg_zmean_window = None

    @staticmethod
    def batch_get(batch_size):

        results = []
        for i in range(batch_size):
            results.append(frame_info.get(block=True))

        return results

    @staticmethod
    def normalize_signal(signal):
        signal = signal/np.mean(signal)
        return signal

    # Finds highest frequency in hz
    @staticmethod
    def find_highest_freq(signal):
        freq_hz = abs(fftfreq(signal.shape[0]) * 30)
        max_idx = np.argmax(signal)
        max_freq = freq_hz[max_idx]
        return max_freq

    def process_data(self):
        while True:
            results = SignalProcessor.batch_get(self.frames_in_window)
            start = perf_counter()
            print("Received "+str(self.frames_in_window)+ " at "+str(start))
            self._process_window(results)
            print("Processed "+str(self.frames_in_window)+ " in "+str(perf_counter()-start))

    def _process_window(self, window):
        self.frame_timestamps_window, self.frame_ids_window, self.rgb_values_window, self.rpy_values_window = zip(*window)
        self.red_window, self.green_window, self.blue_window = zip(*self.rgb_values_window)
        self.roll_window, self.pitch_window, self.yaw_window = zip(*self.rpy_values_window)

        self._resample()
        self._apply_pos()
        self._apply_signal_filtering()
        self._apply_post_processing()
        self._append_window_signals()

    def _resample(self):
        frame_collected_vector = range(self.frames_in_window)
        time_elapsed = self.frame_timestamps_window[-1]-self.frame_timestamps_window[0]
        self.adj_frames_in_window = int(time_elapsed*self.fps)
        frame_limit_vector = range(self.adj_frames_in_window)
        # RGB
        self.red_window = np.interp(frame_limit_vector, frame_collected_vector,self.red_window)
        self.green_window = np.interp(frame_limit_vector, frame_collected_vector,self.green_window)
        self.blue_window = np.interp(frame_limit_vector, frame_collected_vector,self.blue_window)

        # RPY
        self.roll_window = np.interp(frame_limit_vector, frame_collected_vector,self.roll_window)
        self.pitch_window = np.interp(frame_limit_vector, frame_collected_vector,self.pitch_window)
        self.yaw_window = np.interp(frame_limit_vector, frame_collected_vector,self.yaw_window)

    def _apply_pos(self):
        self.red_window = self.normalize_signal(self.red_window)
        self.green_window = self.normalize_signal(self.green_window)
        self.blue_window = self.normalize_signal(self.blue_window)

        mat = np.array([self.red_window, self.green_window, self.blue_window])
        mean_color = np.mean(mat, axis=1)
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        mat_n = np.matmul(diag_mean_color_inv,mat)
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        self.S_window = np.matmul(projection_matrix,mat_n)

        std = np.array([1,np.std(self.S_window[0,:])/np.std(self.S_window[1,:])])
        self.P_window = np.matmul(std,self.S_window)
        self.rppg_window = self.P_window-np.mean(self.P_window)

    def _apply_signal_filtering(self):
        self._apply_rmns()
        self._apply_wnb_filter()

    def _apply_rmns(self):
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

        # uncomment to get rmns
        #self.rppg_fft_rmns_window = self.rppg_fft_window - self.combined_rpy_fft_window
        self.rppg_fft_rmns_window = self.rppg_fft_window

    def _apply_wnb_filter(self):
        bandwidth = .2
        nyq = 0.5 * 30
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
        self.rppg_freq_filtered_window = np.abs(ifft(self.rppg_fft_rmns_window))

    def _apply_post_processing(self):
        self.rppg_zmean_window = (self.rppg_filtered_window - np.mean(self.rppg_filtered_window))/np.std(self.rppg_filtered_window)
        shift_val = 1
        #take shift_val out of the queue
        SignalProcessor.batch_get(shift_val)

    def _append_window_signals(self):
        print('Appended window signals')
        # RGB signals

        for i in range(self.adj_frames_in_window):
            red.put(self.red_window[i])
            green.put(self.green_window[i])
            blue.put(self.blue_window[i])

            roll.put(self.roll_window[i])
            pitch.put(self.pitch_window[i])
            yaw.put(self.yaw_window[i])

            S1.put(self.S_window[0, i])
            S2.put(self.S_window[1, i])
            P.put(self.P_window[i])

            rppg.put(self.rppg_window[i])

            roll_fft.put(self.roll_fft_window[i])
            pitch_fft.put(self.pitch_fft_window[i])
            yaw_fft.put(self.yaw_fft_window[i])

            rppg_fft.put(self.rppg_fft_window[i])

            combined_rpy_fft.put(self.combined_rpy_fft_window[i])
            rppg_fft_rmns.put(self.rppg_fft_rmns_window[i])

            rppg_filtered.put(self.rppg_filtered_window[i])
            rppg_zmean.put(self.rppg_zmean_window[i])

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

class SignalPlotter:

    def __init__(self):
        self.app = dash.Dash(__name__)
        self.app.callback(Output('rppg-plotter', 'figure'),
                          Output('rppg-filtered-plotter', 'figure'),
                          Output('rgb-plotter', 'figure'),
                          Output('rpy-plotter', 'figure'),
                          Output('rppg-zmean-plotter', 'figure'),
                          Input('graph-update', 'n_intervals')
                          )(self.update_graphs)

    @staticmethod
    def get_df():
        x = list(range(len(list(red.queue))))
        red_ = list(red.queue)
        green_ = list(green.queue)
        blue_ = list(blue.queue)
        roll_ = list(roll.queue)
        pitch_ = list(pitch.queue)
        yaw_ = list(yaw.queue)
        rppg_ = list(rppg.queue)
        rppg_filtered_ = list(rppg_filtered.queue)
        rppg_zmean_ = list(rppg_zmean.queue)
        x_zmean = list(range(len(rppg_zmean_)))

        cols = ['frame_id', 'red', 'green', 'blue', 'roll', 'pitch', 'yaw', 'rppg', 'rppg_filtered']
        data = list(zip(x, red_, green_, blue_, roll_, pitch_, yaw_, rppg_, rppg_filtered_))
        df = pd.DataFrame(data, columns=cols)

        cols1 = ['frame_id', 'rppg_zmean']
        data1 = list(zip(x_zmean, rppg_zmean_))
        df1 = pd.DataFrame(data1, columns=cols1)

        return df, df1

    def update_graphs(self, n):
        df, df1 = self.get_df()
        fig0 = px.scatter(df, x="frame_id", y=['rppg'],template="seaborn")
        fig0.update_traces(mode='lines',showlegend=True)
        fig1 = px.scatter(df, x="frame_id", y=['rppg_filtered'],template="seaborn")
        fig1.update_traces(mode='lines',showlegend=True)
        fig2 = px.scatter(df, x="frame_id", y=["red", "green", "blue"],template="seaborn", color_discrete_sequence=["red", "green", "blue"])
        fig2.update_traces(mode='lines',showlegend=True)
        fig3 = px.scatter(df, x="frame_id", y=["roll", "pitch", "yaw"],template="seaborn")
        fig3.update_traces(mode='lines', showlegend=True)
        fig4 = px.scatter(df1, x="frame_id", y=["rppg_zmean"],template="seaborn")
        fig4.update_traces(mode='lines', showlegend=True)

        return fig0, fig1, fig2, fig3, fig4

    def plot(self):

        fig0, fig1, fig2, fig3, fig4 = self.update_graphs(0)

        self.app.layout = html.Div(children=[
            html.H2(children='Signal Plotter'),
            html.H3(children='RPPG Signal'),
            dcc.Graph(
                id='rppg-plotter',
                figure=fig0
            ),
            html.H3(children='RPPG Filtered Signal'),
            dcc.Graph(
                id='rppg-filtered-plotter',
                figure=fig1
            ),
            html.H3(children='RGB Signal'),
            dcc.Graph(
                id='rgb-plotter',
                figure=fig2
            ),
            html.H3(children='RPY Signal'),
            dcc.Graph(
                id='rpy-plotter',
                figure=fig3
            ),
            html.H3(children='RPPG z-mean Signal'),
            dcc.Graph(
                id='rppg-zmean-plotter',
                figure=fig4
            ),
            dcc.Interval(
                id='graph-update',
                interval=1000,
                n_intervals=0
            ),
        ])
        self.app.run_server(debug=False)


def streamer():
    predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
    filename = "./data.avi"
    fs = FaceStreamer(predictor_path, filename=filename)
    fs.stream()


def processor():
    sig_processor = SignalProcessor()
    sig_processor.process_data()


def plotter():
    sig_plotter = SignalPlotter()
    sleep(15)
    sig_plotter.plot()


if __name__ == "__main__":

    t1 = threading.Thread(target=plotter)
    t2 = threading.Thread(target=processor)
    t1.start()
    t2.start()
    streamer()

