from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.epsilon_profile import EpsilonProfile
from controller.random_agent import RandomAgent
from controller.qlearning import QAgent

def main():

    game = SpaceInvaders(display=True)
    gamma = 0.5
    alpha = 0.8
    eps_profile = EpsilonProfile(1.0, 0.1) #probabilité d'exploiration
    #controller = KeyboardController()
    controller = QAgent(game,eps_profile,gamma,alpha)
    #controller = RandomAgent(game.na)
 
    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)

if __name__ == '__main__' :
    main()
