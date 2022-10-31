#!/usr/bin/python3 #shebang

__author__ = "Ivan Zhezhera"
__company__ = "it-enterprise"
__copyright__ = "None"
__license__ = "None"
__version__ = "1.0"
__maintainer__ = "None"
__email__ = "zhezhera@it-enterprise.com"
__category__  = "Product"
__status__ = "Development"

from flask import Flask, render_template, Response, request, redirect, jsonify
from flask import send_from_directory
from threading import Thread, Event

import socket
import requests
import os
import sys
import json
import camera as cm


my_var = [0]
msg_full = [0]

stop_threads = False
stop_threads2 = False

flag = [0]

class HighVision:
    """***Parent class  HighVision"""
    def __init__(self):
        pass
       
                     

class ArtificialIntelligenceSystem(HighVision):
    """Child class ArtificialIntelligenceSystem. Include capability of video data processing with AI, 
    machine and deep learning"""
    def __init__(self):
        self.host = settings['connection']['host']
        self.tcp_flask_port = settings['connection']['tcp_flask_port']
        self.tcp_data_port_1 = settings['connection']['tcp_data_port_1']

        BridgeCommunication.__init__(self)


    def videoProcessing_web(self, settings, var, conn = 0, mode = 0): 

        if (mode == 0):
            #wi.videoToIp(host = self.host, tcp_flask_port = self.tcp_flask_port) 

            cm.VideoCamera().videoToWeb(host = self.host, tcp_flask_port = self.tcp_flask_port, settings = settings) 
            #cm.VideoCamera().get_all_frames_pylon(settings = settings)


 
class BridgeCommunication(HighVision):
    """docstring for ClassName"""

    def __init__(self):
        self.host = settings['connection']['host']
        self.tcp_flask_port = settings["connection"]["tcp_flask_port"]
        self.tcp_data_port_1 = settings["connection"]["tcp_data_port_1"]

    def Transmition(self, data):
        try: #<<<<<<<<<<<<<<<<<<<<<<<
            req_msg = conn.recv(1024).decode('utf-8')#<<<<<<<<<<<<<<<<<<<<<<<
            print(req_msg + str(" "))
            if (req_msg == 'get'):#<<<<<<<<<<<<<<<<<<<<<<<
                print("really get")
                conn.sendall((str(len(data))+str(",")+' '.join(map(str, data))).encode('utf-8'))
            elif(req_msg == 'stop'):#<<<<<<<<<<<<<<<<<<<<<<<                            
                sys.exit(0)

            else:
                conn.close()
                raise Exception
                sys.close()
        except:#<<<<<<<<<<<<<<<<<<<<<<<
            #thread1.join()
            #thread2.join()
            raise Exception

    def command_reaciving(self, conn): 
        while True:
            if (conn):
            #with conn:
                data = conn.recv(1)
                if not data:
                    break

                elif (data == b's'):
                    data = 0
                    print(" * Local server is starting with a web shell")

                    try:
                        #thread1 = Thread(target = AIS.videoProcessing_web, args = (settings, my_var, conn, 0,))
                        #thread2 = Thread(target = BC.Transmition, args = (my_var,))
                        thread1 = Thread(target = AIS.videoProcessing_web(settings = settings, var = my_var, conn = conn, mode = 0))
                        thread1.start()
                        #thread2.start()

                    except KeyboardInterrupt:
                        thread1.join()
                        #break

                    except:
                        print(" ! Except in threads")
                        thread1.join()
                        #thread2.join()

                else:
                    print(' ! Uncorrect letter')





if __name__ == "__main__": 

    global settings

    with open("./settings.json", "r") as read_file:
        settings = json.load(read_file)

    # Init of each classes
    app = Flask(__name__)
    HV = HighVision()
    AIS = ArtificialIntelligenceSystem()
    BC = BridgeCommunication()
    print(" * Initialithation is done") 

    # server initialization
    
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((settings['connection']['host'], settings['connection']['tcp_data_port_1']))
    serversocket.listen(5)
    print(" * Starting SMART VISION SYSTEM")



    # main loop
    while True:
        try: 
            print(" * Ready for the command")
            conn, addr = serversocket.accept()

            # getting the command
            BC.command_reaciving(conn)

        except KeyboardInterrupt:
            print(" ! Global keyboard except") 
            sys.exit(0) 

        except:
            # close client connection
            print(" ! Global except. Check also camera connection to the LAN... ") 
            #sys.exit(0)
 
