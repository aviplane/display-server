import sys  # nopep8
sys.path.append("X:\\")
import labscript_utils.h5_lock
sys.path.append("S:\\Schleier Lab Dropbox\\Cavity Lab Data\\Cavity Lab Scripts\\cavity_analysis")  # nopep8
import readFiles as rf
import AnalysisFunctions as af
import time
import zprocess
from collections import Iterable
from time import sleep
import h5py
import numpy as np
from pymba import Vimba
import random
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont
from PyQt5.QtNetwork import *
import pyqtgraph as pg
from matplotlib import cm
from PIL import Image
import glob
# ccheck_version('zprocess', '1.3.3', '3.0')
import zmq
import scipy.ndimage.measurements as imagemeas
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5

from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

import matplotlib.patches as patches

class Server(QObject):
    plot_made = pyqtSignal(int, np.ndarray)
    ixon_plot = pyqtSignal(np.ndarray)
    shot_name = pyqtSignal(str)

    def __init__(self, port, parameters):
        super(Server, self).__init__()
        self.port = str(port)
        self.parameters = parameters

    @pyqtSlot()
    def make_plot(self, param, data):
        self.plot_made.emit(self.parameters.index(param), np.squeeze(data).T)

    def mainloop(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % port)
        while True:
            #  Wait for next request from client
            message = socket.recv().decode("utf-8")
            print("Received request: ", message)
            if message != "":
                socket.send_string('ok')
                out = self.process_request(message)
            else:
                socket.send_string('done')

    def process_request(self, request_data):
        try:
            print(request_data)
            if request_data == 'hello':
                return 'ok'
            elif request_data.endswith('.h5'):
                self._h5_filepath = request_data
                self.transition_to_buffered(self._h5_filepath)
                return 'done'
            elif request_data == 'done':
                plot_thread = threading.Thread(target=self.transition_to_static,
                                               args=(self._h5_filepath, ))
                plot_thread.start()
                self._h5_filepath = None
                return 'done'
            elif request_data == 'abort':
                self.abort()
                self._h5_filepath = None
                return 'done'
            else:
                raise ValueError('invalid request: %s' % request_data)
        except Exception:
            if self._h5_filepath is not None and request_data != 'abort':
                try:
                    self.abort()
                except Exception as e:
                    sys.stderr.write(
                        'Exception in self.abort() while handling another exception:\n{}\n'.format(str(e)))
            self._h5_filepath = None
            raise

    def transition_to_buffered(self, h5_filepath):
        print('transition to buffered')

    @pyqtSlot()
    def transition_to_static(self, h5_filepath):
        sleep(2.6)
        for param in self.parameters:
            print(f"Getting Data for {param}")
            try:
                data = rf.getdata(h5_filepath, param)
            except Exception as e:
                data = [[1, 0], [0, 1]]
                traceback.print_exc()
                print("%s is not a valid parameter" % param)
            self.make_plot(param, data)
        try:
            ixon_data = rf.getdata(h5_filepath, "ixonatoms")
            self.ixon_plot.emit(ixon_data)
        except Exception as e:
            self.ixon_plot.emit(np.random.randn(5, 5, 3))
            print(e)
        ### Set shot_name Name ###
        try:
            head, tail = os.path.split(h5_filepath)
            with h5py.File(h5_filepath) as hf:
                n_runs = hf.attrs['n_runs']
                n = hf.attrs['run number']
                rep = hf.attrs['run repeat'] if 'run repeat' in hf.attrs else 0
            variables, values, units = af.get_xlabel_single_shot(h5_filepath)
            #print(variables, values)
            for value in values:
                if isinstance(value, Iterable):
                    value = np.array(value)
            variable_string = "\t".join(
                ["{} - {} {}".format(variable, value * sf, name) for variable, value, (sf, name) in
                 zip(variables, values, units)])
            shot_string = "{}\n{}/{} - rep {}\n{}".format(tail,
                                                          n + 1,
                                                          n_runs,
                                                          rep,
                                                          variable_string)
            self.shot_name.emit(shot_string)
        except Exception as e:
            print("Error while generating shot name!")
            print(e)
        print('transition to static')

    def abort(self):
        print('abort')

from scipy import ndimage as ndi

class DisplayServer(QWidget):
    def __init__(self, port):
        super(DisplayServer, self).__init__()
        self.setWindowTitle("Display Server")
        self._h5_filepath = None
        # Generate colormaps
        self.set_colors()
        # Default parameters
        self.parameters = ['Manta223_MOTatoms', 'GreyCavityTransmissionBare', 'roiSum']
        self.server = Server(port=port, parameters=self.parameters)
        # Server GUI interaction
        self.server.plot_made.connect(self.update_plots)
        self.server.ixon_plot.connect(self.ixon_plot)
        self.server.shot_name.connect(self.set_shot_name)
        # Start server thread
        self.thread = QThread()
        self.server.moveToThread(self.thread)
        self.thread.started.connect(self.server.mainloop)
        self.thread.start()
        self.trap_location = np.load(
            "G:/Shared drives/Cavity Drive/avikar/display_server/trap_location.npy")
        self.layout = QGridLayout()
        self.set_up_gui()

        self.setLayout(self.layout)
        self.image = np.flip(np.asarray(Image.open('C:\\Users\\QuantumEngineer\\Pictures\\otter.jpg')),
                             axis=0)

    def set_colors(self):
        colormap = cm.get_cmap("jet")  # cm.get_cmap("CMRmap")
        colormap._init()
        self.lut = (colormap._lut).view(np.ndarray)
        pos = np.linspace(0, 1, len(self.lut))
        self.cmap = pg.ColorMap(pos[:-2], self.lut[:-2])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.update_layout()

    def set_up_gui(self):
        # void QGridLayout::addWidget(QWidget *widget, int fromRow, int fromColumn, int rowSpan, int columnSpan, Qt::Alignment alignment = Qt::Alignment())
        self.parameter_label = QLabel("Parameters: ")
        self.parameter_input = QLineEdit()
        self.parameter_input.setText(', '.join(self.parameters))
        self.shot_name = QLabel("Shot Name: ")
        # self.parameter_input.editingFinished.connect(self.update_layout)
        self.layout.addWidget(self.shot_name, 0, 0, 1, 5)
        self.layout.addWidget(self.parameter_label, 1, 0)
        self.layout.addWidget(self.parameter_input, 1, 1, 1, 3)
        self.titles = [QLabel(i) for i in self.parameters]
        self.mins = [QLabel("Min: ") for i in self.parameters]
        self.maxs = [QLabel("Max: ") for i in self.parameters]
        # Make text big so everybody can read it
        self.font = QFont()
        self.font.setPointSize(24)
        for i in range(len(self.parameters)):
            self.mins[i].setFont(self.font)
            self.maxs[i].setFont(self.font)
            self.titles[i].setFont(self.font)
        self.shot_name.setFont(self.font)
        # Add titles to layout
        for i, title in enumerate(self.titles):
            self.layout.addWidget(title, 2 * i + 2, 0)
            self.layout.addWidget(self.mins[i], 2 * i + 2, 2)
            self.layout.addWidget(self.maxs[i], 2 * i + 2, 3)
            self.layout.addWidget(
                pg.GraphicsLayoutWidget(), 2 * i + 3, 0, 1, 4)
            plot = self.get_random_dog()
            widgetToUpdate = self.layout.itemAtPosition(2 * i + 3, 0).widget()
            try:
                widgetToUpdate.clear()
                widgetToUpdate.show()
                a = widgetToUpdate.addPlot()
                a.enableAutoScale()
                a.setClipToView(True)
                a.showGrid(x=True, y=True)
                a.addItem(plot)
            except Exception as e:
                print(e)

        self.plot_widget = FigureCanvas(
            Figure(figsize=(4, 4), tight_layout=True))
        self._plot_ax = self.plot_widget.figure.subplots()
        self.layout.addWidget(self.plot_widget, 2, 4, len(self.titles) * 2, 2)
        self.ixon_min = QLineEdit()
        self.ixon_max = QLineEdit()
        self.ixon_max.setText('600')
        self.ixon_min.setText('500')

        self.layout.addWidget(self.ixon_min, 1, 4, 1, 1)
        self.layout.addWidget(self.ixon_max, 1, 5, 1, 1)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)
        self.layout.setColumnStretch(3, 1)
        self.layout.setColumnStretch(4, 2)
        self.layout.setColumnStretch(5, 2)

    @pyqtSlot(int, np.ndarray)
    def update_plots(self, num, data):
        plot, mi, ma = self.make_plot(num, data)
        self.mins[num].setText("Min: {:.2f}".format(mi))
        self.maxs[num].setText("Max: {:.2f}".format(ma))
        widgetToUpdate = self.layout.itemAtPosition(2 * num + 3, 0).widget()
        widgetToUpdate.clear()
        widgetToUpdate.show()
        a = widgetToUpdate.addPlot()
        a.enableAutoScale()
        a.setClipToView(True)
        a.showGrid(x=True, y=True)
        a.addItem(plot)
        return

    def rotate_point(self, x, y, x0, y0, angle):
        theta = np.radians(angle)

        r = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta),  np.cos(theta)) ))
        r_vec = r @ np.array([x - x0, y - y0])
        return np.array([x0, y0]) + r_vec

    @pyqtSlot(np.ndarray)
    def ixon_plot(self, data, plot_roi: bool=True):
        vmin = self.ixon_min.text()
        vmax = self.ixon_max.text()
        vmin = None if vmin == '' else int(vmin)
        vmax = None if vmin == '' else int(vmax)
        vmax = max(vmin + 1, vmax)
        self._plot_ax.cla()
        if plot_roi:
            rotangle=-45.5  # in degrees

            numStates=5
            num_states = numStates

            # rotangle, xshift, yshift, length, height, x_start, y_start = af.rotangle, af.xshift, af.yshift, af.length, af.height, af.x_start, af.y_start #lol w h y
            rotangle, xshift, yshift, length, height, x_start, y_start, remaining_offset, lattice_height, lattice_center = af.load_roi_parameters()
            n_traps, trap_width, trap_start, trap_distance = af.load_trap_parameters()
            bottomROIx, bottomROIy = np.array([x_start, x_start + length]), np.array([y_start, y_start + height])

            xpts = np.floor(np.array([bottomROIx + i * xshift  for i in range(num_states)]))
            ypts = np.floor(np.array([bottomROIy + i * yshift - (i > 0) * remaining_offset for i in range(num_states)]))
            xROIlen, yROIlen=np.ptp(bottomROIx), np.ptp(bottomROIy)

            # image=ndi.rotate(data, rotangle, order=0)
            self._plot_ax.imshow(data.squeeze(), aspect="auto", vmin=vmin,
                                 vmax=vmax, interpolation=None)


            # State rectangles
            rects=[patches.Rectangle(self.rotate_point(xpts[i, 0], ypts[i, 0], xpts[i, 0] + xROIlen/2,  ypts[i, 0] + yROIlen/2, rotangle), xROIlen, yROIlen,
                                     linewidth=1, angle = rotangle, ls = '--', edgecolor='white', facecolor='none', alpha = 0.5)
                   for i in range(num_states)]

            # lattice lines
            for i in range(num_states):
                center =  xpts[i, 0] + xROIlen/2,  ypts[i, 0] + yROIlen/2
                high_x_l, high_y_l = self.rotate_point(xpts[i, 0], ypts[i, 0],
                                     *center, rotangle)
                high_x_r, high_y_r = self.rotate_point(xpts[i, 0]+ xROIlen, ypts[i, 0],
                                      *center, rotangle)
                low_x_l, low_y_l = self.rotate_point(xpts[i, 0], ypts[i, 0] + yROIlen,
                                     *center, rotangle)
                low_x_r, low_y_r = self.rotate_point(xpts[i, 0]+ xROIlen, ypts[i, 0]+ yROIlen,
                                      *center, rotangle)

                rot_height = abs(high_y_r - low_y_l)

            #     ax.plot([low_x_l, low_x_r], np.array([low_y_l, low_y_r]), c = 'w', ls = '-')
                self._plot_ax.plot([high_x_l, high_x_r], np.array([high_y_l, high_y_r]) + (lattice_center - lattice_height/2)/rot_height * height, c = 'w', ls = '-', alpha = 0.4)
                self._plot_ax.plot([high_x_l, high_x_r], np.array([high_y_l, high_y_r]) + (lattice_center + lattice_height/2)/rot_height * height, c = 'w', ls = '-', alpha= 0.4)

                for trap in np.arange(n_traps, dtype = int):
                    y_slope = (high_y_l - high_y_r)/n_traps
                    self._plot_ax.plot([high_x_l + trap_start + trap * trap_distance, low_x_l + trap_start + trap * trap_distance],
                            np.array([high_y_l, low_y_l]) - y_slope * trap, c="mintcream", ls = ":", alpha = 0.4)

            for r in rects:
                self._plot_ax.add_patch(r)

        self._plot_ax.set_ylim(1024, 0)
        self._plot_ax.set_title(np.max(data))
        self._plot_ax.figure.canvas.draw()


    @pyqtSlot(str)
    def set_shot_name(self, string):
        self.shot_name.setText(string)

    def make_plot(self, num, data):
        param = self.parameters[num]
        print(f"Plotting {param}")
        try:
            if 'Manta' in param:
                data = np.squeeze(data)
                j = pg.ImageItem()
                max_val = np.max(data)
                min_val = np.min(data)
                if '223_MOT' in param:
                    max_val = np.max(data[100:300, 100:300])
                if '145_MOT' in param:
                    data = data[600:1000, 300:800]
                if '223_Trap' in param:
                    data = np.int32(data.squeeze())
                    data = data[600:800, 400:500]
                    max_val = np.max(data)
                if '145_Raman' in param:
                    data = np.clip(data, 14, None) - 14
                    min_val, max_val = imagemeas.center_of_mass(data)
                try:
                    print(np.min(data), np.max(data), param)
                    j.setImage(
                        image=data)
                    j.setLevels([np.min(data) - 5, np.max(data) + 5])
                except Exception as e:
                    print(e)
                    j = self.get_random_dog()
                    return j, 0, 0
                return j, min_val, max_val
            if 'SPCM' in param:
                dat = np.diff(data)
                i = pg.PlotDataItem(dat)
                return i, np.min(dat), np.max(dat)
            if 'Rigol' in param or 'ProbeLock' in param:
                print(f"Plotting Rigol Data = {data} with length {len(data)}")
                data = data[:-1]  # last point is usually strange/an artifact
                data = np.round(np.array(data), 8)
                i = pg.PlotDataItem(data)
                return i, np.min(data), np.max(data)
            if "ProbeErr" in param:
                data = [i[1] for i in data]
                i = pg.PlotDataItem(data)
                return i, np.min(data), np.max(data)
            elif 'Cavity' in param:
                data = [i[1] for i in data]
                i = pg.PlotDataItem(data)
                return i, np.min(data), np.max(data)
            if 'roi' in param:
                #data = np.array([data]).T
                #levels = (0, np.max(data) + 5)
                if "Magnetization" in param:
                    levels = (-1, 1)
                j = pg.PlotDataItem(data)
                # pg.ImageItem()
                #j.setImage(image=data, levels=levels, lut=self.cmap.getLookupTable(mode='float'))
                # j.setLookupTable(self.cmap.getLookupTable())
                return j, np.min(data), np.mean(data[:800])  # np.max(data)
        except Exception as e:
            traceback.print_exc()
            j = self.get_random_dog()
            return j, 0, 0
        return

    def get_random_dog(self):
        path = 'C:\\Users\\QuantumEngineer\\Pictures\\'
        images = glob.glob(path + "*.jpg")
        self.image = np.flip(np.asarray(Image.open(random.choice(images))),
                             axis=0)
        j = pg.ImageItem(self.image, axisOrder='row-major')
        return j

    def update_layout(self):
        parameter_list = self.parameter_input.text()
        self.parameters = [i.strip() for i in parameter_list.split(',')]
        self.server.parameters = self.parameters
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().deleteLater()
        self.set_up_gui()


if __name__ == '__main__':
    port = 22552
    print('Starting display server on port %d...' % port)
    from requests import get

    data = get('https://ipapi.co/ip/').text
    print('Public IP: ' + data)
    app = QApplication([])
    window = DisplayServer(port)
    window.show()
    app.exec_()
