import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from othello import OthelloGame

MODEL_PATH = "final_model.keras"


def move_to_index(row,col):
    return row * 8+ col

def board_to_tensor(board, current_player):

    canonical = board *current_player

    my_stones = (canonical ==1).astype(np.float32)
    opp_stones = (canonical ==-1).astype(np.float32)

    tensor = np.stack([my_stones, opp_stones], axis=-1)


    return np.expand_dims(tensor,axis =0)



    
# If a previously trained model exists, keep training it instead of starting
# over from random weights -- otherwise build and compile a fresh network.
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"loaded existing model from {MODEL_PATH}, continuing training")
else:
    board_input = layers.Input(shape=(8,8,2))

    first_layer = layers.Conv2D(activation ="relu", filters =64, kernel_size=3, padding="same")(board_input)
    second_layer = layers.Conv2D(activation="relu", filters =64, kernel_size=3, padding="same")(first_layer)
    third_layer = layers.Conv2D(activation = "relu", filters =64, kernel_size=3, padding="same")(second_layer)

    flattened = layers.Flatten()(third_layer)

    policy = layers.Dense(64, activation = "softmax", name="policy")(flattened)
    value = layers.Dense(1, activation = "tanh", name = "value")(flattened)

    model = models.Model(inputs=board_input, outputs=[policy,value])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  metrics={
                      "policy": "accuracy",
                      "value": "mae",
                  },
                  loss={
                      "policy": tf.keras.losses.CategoricalCrossentropy(),
                      "value": tf.keras.losses.MeanSquaredError()
                  }

    )
    print(f"no existing model at {MODEL_PATH}, starting fresh")

class Node:
    def __init__(self,current_player,prior,parent):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.current_player = current_player
        self.parent = parent

    def mean_value(self):
        if self.visit_count ==0:
            Q=0
        else:
            Q = self.value_sum/self.visit_count
        return Q 
    def expand(self,game):
        self.legal_moves = game.legal_moves()
        self.current_position = game.board
        current_position_tensor = board_to_tensor(self.current_position,self.current_player)
        policy_output, value_output = model.predict(current_position_tensor)
        policy_output_flat = policy_output[0]
        for i in self.legal_moves:
            row,col = i
            index = move_to_index(row,col)
            prior = policy_output_flat[index]
            self.children[i] = Node(self.current_player *-1,prior,self)
        return value_output[0][0]
    def best_child(self):
        c_puct = 1.0
        best_score = None
        best_move= None
        for i in self.children.items():
            move,child= i
            score = (-1 * child.mean_value()) + c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            if best_score == None:
                best_score = score
                best_move = move
            if score>best_score:
                best_score = score
                best_move = move
        return best_move, best_score
    def backup(self,value):
        self.visit_count +=1
        self.value_sum +=value
        if self.parent != None:
            self.parent.backup(-value)
    def select_final_move(self):
        final_move = None
        final_move_count = None
        for i in self.children.items():
            move , child = i
            if final_move_count == None:
                final_move = move
                final_move_count = child.visit_count
            if final_move_count <  child.visit_count:
                final_move = move
                final_move_count = child.visit_count
        return final_move
    def total_visits(self):
        total_count = 0
        for i in self.children.items():
            move , child = i
            total_count += child.visit_count
        return total_count
    def visit_distribution(self):
        distribution = np.zeros(64)
        total = self.total_visits()
        for i in self.children.items():
            move, child = i 
            index = move_to_index(*move)
            distribution[index] = child.visit_count/total
        return distribution



def run_mcts(game, num_simulations):
    taco = Node(game.current_player,0,None)
    for i in range(num_simulations):
        current = taco
        game_copy = game.copy()
        while len(current.children) != 0:
            move , score = current.best_child()
            current = current.children[move]
            game_copy.play(*move)
        expand = current.expand(game_copy)
        current.backup(expand)
    return taco


#play the actuallly game

def play_self_play_game(num_simulations):
    """Play one full game, MCTS choosing every move for both sides.

    Returns (training_data, winner) where training_data is a list of
    (board_tensor, policy_distribution, value) tuples -- one per position
    that was actually played, value already backfilled with the real
    outcome from that position's own current_player perspective.
    """
    game = OthelloGame()
    game_history = []

    while not game.game_over:
        sim = run_mcts(game, num_simulations)
        final_move = sim.select_final_move()
        game_history.append((
            board_to_tensor(game.board, game.current_player),
            sim.visit_distribution(),
            game.current_player,
        ))
        game.play(*final_move)

    winner = game.winner()
    training_data = []
    for board, dist, current_player in game_history:
        if current_player == winner:
            value = 1
        elif winner == 0:
            value = 0
        else:
            value = -1
        training_data.append((board, dist, value))

    return training_data, winner


def train_on_game(training_data):
    """Stack one game's examples into batches and run one training step.

    Each recorded board already carries its own batch-of-1 dimension (from
    board_to_tensor's expand_dims), so np.concatenate along axis 0 stacks
    them into one real batch instead of nesting an extra dimension.
    """
    boards = np.concatenate([board for board, dist, value in training_data], axis=0)
    policy_targets = np.array([dist for board, dist, value in training_data])
    value_targets = np.array([value for board, dist, value in training_data], dtype=np.float32)

    model.fit(
        boards,
        {"policy": policy_targets, "value": value_targets},
        epochs=1,
        verbose=0,
    )


NUM_GAMES = 100
SIMULATIONS_PER_MOVE = 50
CHECKPOINT_EVERY = 5
LOG_PATH = "training_log.txt"

if __name__ == "__main__":
    with open(LOG_PATH, "a") as log_file:
        for game_num in range(1, NUM_GAMES + 1):
            training_data, winner = play_self_play_game(SIMULATIONS_PER_MOVE)
            train_on_game(training_data)

            winner_name = {1: "black", -1: "white", 0: "draw"}[winner]
            message = f"game {game_num}/{NUM_GAMES}: winner={winner_name}, positions={len(training_data)}"
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

            if game_num % CHECKPOINT_EVERY == 0:
                checkpoint_path = f"checkpoint_game_{game_num}.keras"
                model.save(checkpoint_path)
                save_message = f"  -> saved {checkpoint_path}"
                print(save_message, flush=True)
                log_file.write(save_message + "\n")
                log_file.flush()

        model.save(MODEL_PATH)
        print(f"training complete, saved {MODEL_PATH}", flush=True)

    

























               
#Get the output layer 
# THan we take the output layer to 2 layers that are dense so that sequential model
#One will find the move the other looks at the win rate or likely to go there whatever you want to call it 




