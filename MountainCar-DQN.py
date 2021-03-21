import gym
import numpy as np
import keras

env = gym.make('MountainCar-v0')
n = np.shape(env.observation_space.sample())
num_labels = env.action_space.n

# variables and CONSTANTS
X_train = []
y_train = []
epsilon = 0.6
EPSILON_DECAY = 0.85
TOTAL_GAMES= 100_000
TRAIN_EVERY = 5000
SHOW_RESULT_EVERY = 1000
score_list = []
trained = False

#Build the model
model = keras.Sequential([
    keras.layers.Dense(256, input_dim = n[0]),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_labels, activation='softmax')
])
model.compile(loss="mse", optimizer="adam")

# Training Loop
for game_nr in range(TOTAL_GAMES):
    done=score=0
    obs = env.reset()

    game_obs=[]
    game_action=[]

    while not done:
        if not trained or np.random.uniform() < epsilon: # For exploration
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.asarray([obs]))[0]) # For exploitation

        game_obs.append(obs)
        game_action.append(action)
        obs, reward, done, info = env.step(action)
        score+=reward

    score_list.append(score)

    for obs, act in zip(game_obs, game_action):
        label = np.zeros((num_labels,))
        label[act] = score
        y_train.append(label)
        X_train.append(obs)

    if not (game_nr % SHOW_RESULT_EVERY):
        print(f"{game_nr} / {TOTAL_GAMES}" + \
              f"\tMost recent score: {score}" + \
              f"\tInter-training score-avg: {np.mean(score_list)}")

    if((game_nr+1)%TRAIN_EVERY==0):
        epsilon *= EPSILON_DECAY
        print(f"\nTraining:\tmax score: {np.max(score_list)}\tmean score: {np.mean(score_list):.2f}\tepsilon: {epsilon}")
        # train the Neural Network
        model.fit(
            np.asarray(X_train),
            np.asarray(y_train),
            epochs=5
        )
        trained = True
        X_train = []
        y_train = []
        score_list = []

# Play the game
for _ in range(25):
    done = score = 0
    obs = env.reset()
    while not done:
        action = np.argmax(model.predict(np.asarray([obs]))[0])
        obs, reward, done, info = env.step(action)
        env.render()    # actually render the games
        score += reward
        print(score, end="\r")
    print("")
