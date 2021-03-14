import gym
import numpy as np
import keras

env = gym.make("SpaceInvaders-ram-v0")
n = np.shape(env.observation_space.sample())
num_labels = env.action_space.n

# to save the training data
X_train = []
y_train = []

# variables and CONSTANTS
epsilon = .5
EPSILON_DECAY = 0.85
TOTAL_GAMES = 100000
TRAIN_EVERY = 1000

score_list = []

# build the neural network
model = keras.Sequential([
    keras.layers.Dense(256, input_dim=(128)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])
model.compile(loss="mse", optimizer="adam")

# training loop
for game_nr in range(TOTAL_GAMES):
    done = score = 0
    obs = env.reset()

    # initialize local game_based memory
    game_obs = []
    game_action = []

    while not done:
        obs = obs / 255.0 # normalise
        if not trained or np.random.uniform() < epsilon: # epsilon-decreasing strategy
            action = env.action_space.sample()
        else:
            # action being executed
            action = np.argmax(model.predict(np.asarray([obs]))[0])

        game_obs.append(obs)
        game_action.append(action)
        obs, reward, done, info = env.step(action)
        score += reward

    score_list.append(score)

    for obs, a in zip(game_obs, game_action):
        label = np.zeros((num_labels,))
        label[a] = score
        y_train.append(label)
        X_train.append(obs)


    if not (game_nr%500): # print data after every 500th game
        print(f"{game_nr} / {TOTAL_GAMES}"+\
              f"\tMost recent score: {score}"+\
              f"\tInter-training score-avg: {np.mean(score_list)}")

    if((game_nr+1)%TRAIN_EVERY==0): # train after certain number of games
        epsilon *= EPSILON_DECAY
        print(f"\nTraining:\tmax score: {np.max(score_list)}\tmean score: {np.mean(score_list):.2f}\tepsilon: {epsilon}")
        # train the neural network
        model.fit(
            np.asarray(X_train),
            np.asarray(y_train),
            epochs=5
        )
        trained = True
        X_train = []
        y_train = []
        score_list = []

# play using the trained neural network
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