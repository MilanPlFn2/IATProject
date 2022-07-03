import sys
from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.epsilon_profile import EpsilonProfile
from controller.random_agent import RandomAgent
from controller.qlearning import QAgent
from controller.dqn_agent import DQNAgent
from networks import MLP, CNN

def main(args):

    game = SpaceInvaders(display=False)
    gamma = 1 #coefficient de pondération des récompenses
    alpha = 1 #coefficient de mise à jour
    eps_profile = EpsilonProfile(1.0, 0.1) #probabilité d'exploiration

    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    # Hyperparamètres basiques
    n_episodes = 200
    max_steps = 100
    final_exploration_episode = 1000

    #controller = KeyboardController()
    controller = QAgent(game,eps_profile,gamma,alpha)
    #controller = DQNAgent(model,eps_profile,gamma,alpha)
    #controller = RandomAgent(game.na)

    if(args == "learn"):
        controller.learn(game, n_episodes, max_steps)
        controller.save_qfunction()


    #state = game.reset()
    #while True:
    #    action = controller.select_action(state)
    #    state, reward, is_done = game.step(action)
    #    sleep(0.0001)


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
