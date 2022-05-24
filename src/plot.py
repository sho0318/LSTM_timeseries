nb_epoch = len(MAE_fake)

plt.figure(figsize=(20, 10))

plt.subplot(1,2,2)
plt.xlim(-1,nb_epoch)
plt.plot(range(nb_epoch), MAE_real,     marker='.', label='real')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.xlim(-1,nb_epoch)
plt.plot(range(nb_epoch), MAE_fake, marker='.', label='fake')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.xlim(-1,nb_epoch)
plt.plot(range(nb_epoch), MAE_mix, marker='.', label='mix')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.title("MAE")

nb_epoch = len(loss_real)

plt.subplot(1,2,1)
plt.xlim(-1,nb_epoch)
plt.plot(range(nb_epoch), loss_real,     marker='.', label='real')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.xlim(-1,nb_epoch)
plt.plot(range(nb_epoch), loss_fake, marker='.', label='fake')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.title("MSE")

plt.xlim(-1,nb_epoch)
plt.plot(range(nb_epoch), loss_mix, marker='.', label='mix')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.title("MSE")
plt.show()