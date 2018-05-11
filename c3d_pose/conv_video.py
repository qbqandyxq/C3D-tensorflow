import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
from PIL import ImageEnhance
import random
import numpy as np
import cv2
import time

filename='./output/work.txt'
video_save = '/home/qbq/Desktop/c3d_pose/video_output/'
fps=20

'''
pose_dict = {'0':'ApplyEyeMakeup', '1':'ApplyLipstick', '2':'Archery', '3':'BabyCrawling', '4':'BalanceBeam',
             '5':'BandMarching', '6':'BaseballPitch', '7':'Basketball', '8':'BasketballDunk', '9':'BenchPress',
             '10':'Biking', '11':'Billiards', '12':'BlowDryHair', '13':'BlowingCandles', '14':'BodyWeightSquats',
             '15':'Bowling', '16':'BoxingPunchingBag', '17':'BoxingSpeedBag', '18':'BreastStroke', '19':'BrushingTeeth',
             '20':'CleanAndJerk', '21':'CliffDiving', '22':'CricketBowling', '23':'CricketShot', '24':'CuttingInKitchen',
             '25':'Diving', '26':'Drumming', '27':'Fencing', '28':'FieldHockeyPenalty', '29':'FloorGymnastics',
             '30':'FrisbeeCatch', '31':'FrontCrawl', '32':'GolfSwing', '33':'Haircut', '34':'Hammering',
             '35':'HammerThrow', '36':'HandstandPushups', '37':'HandstandWalking', '38':'HeadMassage', '39':'HighJump',
             '40':'HorseRace', '41':'HorseRiding', '42':'HulaHoop', '43':'IceDancing', '44':'JavelinThrow',
             '45':'JugglingBalls', '46':'JumpingJack', '47':'JumpRope', '48':'Kayaking', '49':'Knitting',
             '50':'LongJump', '51':'Lunges', '52':'MilitaryParade', '53':'Mixing', '54':'MoppingFloor',
             '55':'Nunchucks', '56':'ParallelBars', '57':'PizzaTossing', '58':'PlayingCello', '59':'PlayingDaf',
             '60':'PlayingDhol', '61':'PlayingFlute', '62':'PlayingGuitar', '63':'PlayingPiano', '64':'PlayingSitar',
             '65':'PlayingTabla', '66':'PlayingViolin', '67':'PoleVault', '68':'PommelHorse', '69':'PullUps',
             '70':'Punch', '71':'PushUps', '72':'Rafting', '73':'RockClimbingIndoor', '74':'RopeClimbing',
             '75':'Rowing', '76':'SalsaSpin', '77':'ShavingBeard', '78':'Shotput', '79':'SkateBoarding',
             '80':'Skiing', '81':'Skijet', '82':'SkyDiving', '83':'SoccerJuggling', '84':'SoccerPenalty',
             '85':'StillRings', '86':'SumoWrestling', '87':'Surfing', '88':'Swing', '89':'TableTennisShot',
             '90':'TaiChi', '91':'TennisSwing', '92':'ThrowDiscus', '93':'TrampolineJumping', '94':'Typing',
             '95':'UnevenBars', '96':'VolleyballSpiking', '97':'WalkingWithDog', '98':'WallPushups', '99':'WritingOnBoard',
             '100':'YoYo', 
             }
'''
pose_dict= {'0': 'phone', '1':'smoking'}

def show_video(gt, path, inf):
    
    video_frames = sorted(os.listdir(path))
    gt_ = pose_dict[""+gt+""]
    length = (len(video_frames)//16-1)*16
    a_path = path.split('/')
    video_save_to_dir = os.path.join(video_save, a_path[-2])
    if os.path.isdir(video_save_to_dir) == False:
        os.mkdir(video_save_to_dir)
    
    video_save_to_dir = os.path.join(video_save_to_dir, a_path[-1])
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    videoWriter = cv2.VideoWriter(video_save_to_dir + str('.avi'),fourcc, fps,(1280,720))   
    for i in range(len(inf)):
        
        label = pose_dict[""+inf[i]+""]
        for j in range(i*16, i*16 + 16):
            frame_path = video_frames[j]
            frame = os.path.join(path, frame_path)
            sMatImage = cv2.imread(frame)
       
            font = cv2.FONT_HERSHEY_SIMPLEX
         
            sMatImage = cv2.putText(sMatImage, "inf: "+label+"|=|"+"gt:  "+gt_, (10,25), font, 1,(0,255,0))
            videoWriter.write(sMatImage)
      
    videoWriter.release()
    print("done")    

def main():
    lines = open(filename,'r')
    for line in lines:
        line_ = line.split('|')
        
        gt=line_[0]
        dir_path = line_[1]
        #inf = line_[2][1:-2].strip(',').split(',')
        inf=[a.replace(' ', '') for a in line_[2][1:-2].strip(',').split(',')]
        print("gt",gt)
        print("path",dir_path)
        print("inf", inf)   
        show_video(gt, dir_path, inf)
main()
