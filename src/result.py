class Result():
    def __init__(self,loss_fake,MAE_fake,loss_real,MAE_real,BoxLength,WorkNum):
        self.loss_fake = loss_fake
        self.MAE_fake = MAE_fake
        self.loss_real = loss_real
        self.MAE_real = MAE_real
        self.BoxLength = BoxLength
        self.WorkNum = WorkNum
    
    def mean(x):
        return sum(x)/float(len(x))
    
    def cal_result(self):
        fake_result = []
        real_result = []
        for i in range(self.BoxLength):
            start_num = i * self.WorkNum
            finish_num = (i + 1) * self.WorkNum

            array_fake = [self.mean(self.loss_fake[start_num,finish_num]),self.mean(self.MAE_fake[start_num,finish_num])]
            array_real = [self.mean(self.loss_real[start_num,finish_num]),self.mean(self.MAE_real[start_num,finish_num])]

            fake_result.append(array_fake)
            real_result.append(array_real)
        

        return fake_result,real_result
        
