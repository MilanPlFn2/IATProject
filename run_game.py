from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.epsilon_profile import EpsilonProfile
from controller.random_agent import RandomAgent
from controller.qlearning import QAgent
from controller.dqn_agent import DQNAgent
from networks import MLP, CNN

def main():

    game = SpaceInvaders(display=True)
    gamma = 0.8 #coefficient de pondération des récompenses
    alpha = 0.8 #coefficient de mise à jour
    eps_profile = EpsilonProfile(1.0, 0.1) #probabilité d'exploiration

    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    # Hyperparamètres basiques
    n_episodes = 5000
    max_steps = 50
    final_exploration_episode = 1000

    # Hyperparamètres de DQN
    batch_size = 32
    replay_memory_size = 4000
    target_update_frequency = 100
    tau = 1.0

    model = MLP(game)

    """ INSTANCIE LE RESEAU DE NEURONES """

    #controller = KeyboardController()
    #controller = QAgent(game,eps_profile,gamma,alpha)
    controller = DQNAgent(model,eps_profile,gamma,alpha)
    #controller = RandomAgent(game.na)
 
    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)

if __name__ == '__main__' :
    main()
