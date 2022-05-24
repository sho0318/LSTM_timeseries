import tensorflow as tf
import IPython
import IPython.display

class LSTM_models():
    def __init__(self,window,RealPatience,FakePatience,MAX_EPOCHS):
        self.window = window
        self.MAX_EPOCHS = MAX_EPOCHS
        self.RealPatience = RealPatience
        self.FakePatience = FakePatience

    def compile_and_fit(self,model,train_df,test_df,patience):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(self.window.train(train_df), 
                            epochs=self.MAX_EPOCHS, 
                            validation_data=self.window.test(test_df), 
                            callbacks=[early_stopping])

        return history

    def lstm(self,i,train_df,test_df,train_real):

        OUT_STEPS = 24
        num_features = train_df[0].shape[1]

        fake_lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(OUT_STEPS*num_features,kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        fake_performance = {}
        history = self.compile_and_fit(fake_lstm_model,train_df,test_df,self.FakePatience)

        IPython.display.clear_output()
        fake_performance['LSTM'] = fake_lstm_model.evaluate(self.window.test(test_df))


        real_lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(OUT_STEPS*num_features,kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        real_performance = {}
        history = self.compile_and_fit(real_lstm_model,train_real,test_df,self.RealPatience)

        IPython.display.clear_output()
        real_performance['LSTM'] = real_lstm_model.evaluate(self.window.test(test_df))

        return fake_performance['LSTM'],real_performance['LSTM']



