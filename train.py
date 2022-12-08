from tensorflow.core.framework.dataset_options_pb2 import AUTO
from model import *
import tensorflow as tf
import matplotlib.pyplot as plt

########################################################################################################################
# LOADING DATA
########################################################################################################################


categories, train_data, test_data, x_train, y_train, x_test, y_test = load_data(DATA_PATH)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


########################################################################################################################
# BUILDING MODEL
########################################################################################################################


model = create_model(x_train, x_test, classes=len(categories), trainable_encoder=False)


########################################################################################################################
# TRAINING MODEL
########################################################################################################################


early_stopping = keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max',
                                               patience=30, restore_best_weights=True, verbose=1)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', mode='max',
                                              factor=0.1, patience=10, verbose=1)

csv_logger = keras.callbacks.CSVLogger(os.path.join(TMP_PATH, 'training.csv'))

hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=10,
                 validation_data=(x_test, y_test), verbose=2,
                 callbacks=[early_stopping, reduce_lr, csv_logger])

path = os.path.join(TMP_PATH, 'trained_model.h5')
model.save(path, include_optimizer=False)

plt.clf()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.savefig(os.path.join(TMP_PATH, 'training_loss.png'))

plt.clf()
plt.plot(hist.history['sparse_categorical_accuracy'])
plt.plot(hist.history['val_sparse_categorical_accuracy'])
plt.savefig(os.path.join(TMP_PATH, 'training_accuracy.png'))


########################################################################################################################
# EVALUATE MODEL
########################################################################################################################


res = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(res)

y_out = model.predict(x_test, batch_size=BATCH_SIZE)
y_out = np.argmax(y_out, axis=1)
i = 0
plt.figure(figsize=(4, 4))
for img, out, exp in zip(x_test, y_out, y_test):
    if out != exp:
        plt.clf()
        plt.imshow(img)
        title = '{} misclassified as {}'.format(exp, out)
        plt.title(title)
        i += 1
        plt.savefig(os.path.join(TMP_PATH, '{} ({}).png'.format(i, title)))

# filenames = tf.data.Dataset.list_files(WAV_DATA_PATTERN, seed=35155)  # This also shuffles the images
# dataset_2d = filenames.map(parser_2d, num_parallel_calls=AUTO)
# dataset_2d = dataset_2d.apply(tf.data.experimental.ignore_errors())  # drop examples which are shorter
# dataset_2d = dataset_2d.batch(shard_size)

#export_spectrogram_for_wav(DATA_PATH, WAV_PATH)

# spec = get_melspectrogram_db('D:\\MusicGenreClassificator\\data\\African\\057503.mp3')
# img = spec_to_image(spec)
# plt.imshow(img)
# plt.show()
# print('aaa')

# sound = AudioSegment.from_mp3('D:\\MusicGenreClassificator\\data\\African\\057503.mp3')
# sound.export('D:\\MusicGenreClassificator\\data_wav\\057503.wav', format="wav")
