import os
import datetime
import IPython
import IPython.display
import numpy as np
import pandas as pd
import tensorflow as tf
from data_devied import DataHundle,RealDataHundle
from split_window import WindowGenerator
from lstm import LSTM_models
from result import Result
import pickle
import requests
from local_setting import *

DataNum = 40 #dataの数
MAX_EPOCHS = 100 #epoch数
WorkNum = 50 #work_num回分の平均
PatNum = 10 #earlystoppingのpatience
RealPatience = 10
FakePatience = 1
DataDirectory = "0.7"
BoxLength = 7 #実データを何分割にしているか
with open('../data/real_data/real_data.binaryfile', 'rb') as web:
  RealBox = pickle.load(web)
fake_data=[]

class LINENotifyBot(object):
    API_URL = 'https://notify-api.line.me/api/notify'
    def __init__(self):
        self.__headers = {'Authorization': 'Bearer ' + LINE_ACCESS_TOKEN}

    def send(
        self,
        message
    ):
        payload = {
            'message': message,
        }
        r = requests.post(
            LINENotifyBot.API_URL,
            headers=self.__headers,
            data=payload,
        )

def mean(a,num):
  start_num = num * WorkNum
  finish_num = (num+1) * WorkNum
  return sum(a[start_num:finish_num])/float(WorkNum)

def process_result(a,b,c,d,num):
  return [mean(a,num),mean(b,num),mean(c,num),mean(d,num)]

def main():
  Line_bot = LINENotifyBot()

  datahundle = DataHundle(DataDirectory,DataNum)
  real_datahundle = RealDataHundle(RealBox)

  print("--------------------data deiveid---------------------")

  real_box = real_datahundle.change_realdata()

  for use_data in range(BoxLength):
    fake_data.append(datahundle.read_data(use_data+1))
  
  print("--------------------make models---------------------")

  window = WindowGenerator(input_width=24, label_width=1, shift=24, 
                           train_columns = fake_data[0][0].columns,
                           label_columns=['njobs'], DataNum=DataNum)

  lstm_models = LSTM_models(window,RealPatience,FakePatience,MAX_EPOCHS)

  loss_fake = []
  MAE_fake = []
  loss_real = []
  MAE_real = []

  print("--------------------machine learning---------------------")

  for num in range(len(fake_data)):
    test_df = real_box.pop(0)

    for i in range(WorkNum):
      print(num+1,"分割目")
      print(i+1,"回目")

      fake_loss,real_loss = lstm_models.lstm(num,fake_data[num],test_df,real_box)

      loss_fake.append(fake_loss[0])
      MAE_fake.append(fake_loss[1])
      loss_real.append(real_loss[0])
      MAE_real.append(real_loss[1])
    
    array = process_result(loss_fake,MAE_fake,loss_real,MAE_real,num)
    result_array = pd.DataFrame(array)
    path = "../data/result/process/" + str(num+1) + ".csv"
    result_array.to_csv(path)

    real_box.append(test_df)

    message =str(num+1) + "分割目終了\n" + "fake\n" + str(array[0]) + "\n" + str(array[1]) + "\nreal\n" + str(array[2]) + "\n" + str(array[3])
    Line_bot.send(message)
  
  result = Result(loss_fake,MAE_fake,loss_real,MAE_real,BoxLength,WorkNum)
  fake_result,real_result = result.cal_result()

  fake_result_df = pd.DataFrame(fake_result)
  fake_result_df.columns = ['MSE','MAE']
  real_result_df = pd.DataFrame(real_result)
  real_result_df.columns = ['MSE','MAE']

  fake_result_df.to_csv("../data/result/fakedata.csv")
  real_result_df.to_csv("../data/result/realdata.csv")
 
  message ="\n学習終了"
  Line_bot.send(message)

if __name__ == "__main__":
  main()