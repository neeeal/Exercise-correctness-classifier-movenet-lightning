import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os

main_dir = 'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\keypoints'
exercises = os.listdir(main_dir)

train_mses, train_accuracies = [],[]
test_mses, test_accuracies = [],[]
names = []

main_filepath = f'D:\\Documents\\College\\3rd Year\\2nd Sem\\CPE 020\\final_proj\\models'
files = os.listdir(main_filepath)
current_path = os.path.join(main_filepath,f'models_v{len(files)}')
if os.path.isdir(current_path):
    pass
else:
    os.mkdir(current_path)

for n,ex in enumerate(exercises):
    name = ex.split(".")[0]
    names.append(name)

    print(f'Loading {name} data')
    filepath = os.path.join(main_dir,ex)
    data = pd.read_csv(filepath)
    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                     test_size = 0.20,
                                                     random_state = 42,
                                                     shuffle=True
                                                     )

    print(f'Creating {name} NN Classifier')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=5, min_lr=1e-6)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    inputs = tf.keras.Input(shape=(51,))
    x = Dropout(0.1)(inputs)
    x = BatchNormalization()(x)
    x = Dense(units = 64,
              activation='relu'
              )(x)
    x = Dropout(0.4)(x)
    x = Dense(units = 128,
              activation='relu'
              )(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(units = 256,
              activation='relu'
              )(x)
    x = Dropout(0.4)(x)
    x = Dense(units = 512,
              activation='relu'
              )(x)
    x = Dropout(0.4)(x)
    x = Dense(units = 1024,
              activation='relu'
              )(x)
    x = Dropout(0.4)(x)
    outputs = Dense(units = 1,
                    activation = 'sigmoid')(x)


    model = tf.keras.Model(inputs = inputs,
                           outputs = outputs,
                           name=f'{name}-classifier')
    model.compile(loss='binary_crossentropy',
                  optimizer = Adam(1e-4),
                  metrics = ['acc'])

##    model.summary()

    print('Training...')
    history = model.fit(x = X_train,
                        y = y_train,
                        validation_data = (X_test,y_test),
                        batch_size = 64,
                        epochs = 256,
                        verbose = 0,
                        workers = -1,
                        callbacks=[reduce_lr,es]
                        )
    train_mses.append(history.history['loss'][-1])
    train_accuracies.append(history.history['acc'][-1])

    print('Evaluating...')
    evaluation = model.evaluate(x = X_test,
                                y = y_test,
                                batch_size = 64,
                                verbose = 0)
    test_mses.append(evaluation[0])
    test_accuracies.append(evaluation[1])
    print(f'MSE={evaluation[0]} Accuracy={evaluation[1]}')

    print('Saving...\n')
    filepath = os.path.join(current_path,f'{name}.h5')
    model.save(filepath)
##    if n == 1: break

results = pd.DataFrame(data={'exercises':names,
                             'train_mse':train_mses,
                             'test_mse':test_mses,
                             'train_accuracy':train_accuracies,
                             'test_accuracy':test_accuracies})
print('Training Results')
print(results)
results_path=os.path.join(current_path,f'model_reults.csv')
results.to_csv(results_path,sep=',', index=False, encoding='utf-8')

print('Finished Execution')

