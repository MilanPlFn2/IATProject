import sys
from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.epsilon_profile import EpsilonProfile
from controller.random_agent import RandomAgent
from controller.qlearning import QAgent
from controller.dqn_agent import DQNAgent
from networks import MLP, CNN
import time

def main(args):

    game = SpaceInvaders(display=True)
    gamma = 1 #coefficient de pondération des récompenses
    alpha = 1 #coefficient de mise à jour
    eps_profile = EpsilonProfile(1.0, 0.1) #probabilité d'exploiration

    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    # Hyperparamètres basiques
    n_episodes = 100
    max_steps = 1000

    controller = QAgent(game,eps_profile,gamma,alpha)

    if(args == "learn"):
        controller.learn(game, n_episodes, max_steps)
        controller.save_qfunction()
    elif(args == "test"):
        controller.load_qfunction()
        test_agent(game, controller, 0, True)


def test_agent(game: SpaceInvaders, agent: QAgent, speed: float = 0., display: bool = False):
    n_episodes = 1
    step = 0
    sum_rewards = 0.

    for i in range(n_episodes):
        state = game.reset()
        if (display):
            game.render()
        
        while True:
            action = agent.select_greedy_action(state)
            next_state, reward, terminal = game.step(action)
            step +=1
            if display:
                time.sleep(speed)
                game.render()

            sum_rewards += reward
            if terminal:
                n_steps = step+1  # number of steps taken
                game.game_over()
                break
            state = next_state
    print("perdu, " + str(sum_rewards) + " points")
    return n_steps, sum_rewards

if __name__ == '__main__' :
    if(len(sys.argv) < 2 ): 
        print("")
        print("Manual :")
        print("python3 run_game.py [arg1]")
        print("arg1 :")
        print("     learn : Apprentissage")
        print("     test : Test de l'agent")
        exit(0)
    else:
        main(sys.argv[1])

