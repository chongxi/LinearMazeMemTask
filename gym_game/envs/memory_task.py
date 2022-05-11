import pygame
import math
import numpy as np
import random
import torch.nn.functional as F
import torch

screen_width = 800
screen_height = 250
check_point = {'T1':[412, 471], 'T2':[557, 616]}
begin_location = {'T1': [58, 119.2], 'T2': [58, 57]}
rgb_weight = [0.299, 0.587, 0.114]
speed_limit_low = 8

class Mouse:
    def __init__(self, mouse_file, map_file, pos):
        self.windowWidth, self.windowHeight = screen_width, screen_height
        self.surface = pygame.image.load(mouse_file)
        self.map = pygame.transform.scale(pygame.image.load(map_file), (self.windowWidth,self.windowHeight)).convert()
        self.mouse_body_len = 28
        pygame.draw.line(self.map, 'red', (check_point['T1'][0]+self.mouse_body_len, 0), (check_point['T1'][0]+self.mouse_body_len, 250), 2)
        pygame.draw.line(self.map, 'red', (check_point['T1'][1]+self.mouse_body_len, 0), (check_point['T1'][1]+self.mouse_body_len, 250), 2)
        pygame.draw.line(self.map, 'red', (check_point['T2'][0]+self.mouse_body_len, 0), (check_point['T2'][0]+self.mouse_body_len, 250), 2)
        pygame.draw.line(self.map, 'red', (check_point['T2'][1]+self.mouse_body_len, 0), (check_point['T2'][1]+self.mouse_body_len, 250), 2)
        self.maze_arr = pygame.surfarray.array3d(self.map).swapaxes(0,1) 
        self.surface = pygame.transform.scale(self.surface, (30, 16))
        self.font = pygame.font.SysFont("Arial", 15)
        self.rotate_surface = self.surface
        self.pos = pos
        self.angle = 0
        self.speed = 0
        self.average_speed = 0
        self.licking = False
        self.lick_times = 0
        self.reward = 0  # acquired reward so far in this trial 
        self.center = [self.pos[0] + 15, self.pos[1] + 8] 
        self.radars = []
        self.radars_for_draw = []
        self.trial_finish = False
        self.current_check = 0
        self.prev_distance = 0
        self.cur_distance = 0
        self.goal = False
        self.check_flag = 0
        self.distance = 0
        self.time_spent = 0
        self.obs_arr = None

    def draw(self, screen, trial_type):
        self.trial_type = trial_type
        screen.blit(self.rotate_surface, self.pos)
        ## show animal position in text
        text = self.font.render("pos:{:.2f},{:.2f}   /  Trial Type:{}  / reward:{}".format(self.pos[0], self.pos[1], trial_type, self.reward), True, 'black')
        text_rect = text.get_rect()
        text_rect.center = (120, 10)
        screen.blit(text, text_rect)
        ## show what the animal see
        obs_surf = pygame.surfarray.make_surface(self.obs_arr.swapaxes(0,1))
        screen.blit(obs_surf, (100, 200))

    def draw_collision(self, screen):
        for i in range(4):
            x = int(self.four_points[i][0])
            y = int(self.four_points[i][1])
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 5)

    def draw_radar(self, screen):
        for r in self.radars_for_draw:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_trial_finish(self):
        self.trial_finish = False
        if self.pos[0] >= 670-30:
            self.trial_finish = True

    def check_licking(self):
        self.goal = False
        if self.pos[0] > check_point['T1'][0] and self.pos[0] < check_point['T1'][1] and self.trial_type == 'T1':
            self.check_flag += 1
            self.goal = True
            pygame.draw.circle(self.map, 'green', (int(self.pos[0]+self.mouse_body_len), int(self.pos[1]+8)), 5)
        elif self.pos[0] > check_point['T2'][0] and self.pos[0] < check_point['T2'][1] and self.trial_type == 'T2':
            self.check_flag += 1
            self.goal = True
            pygame.draw.circle(self.map, 'green', (int(self.pos[0]+self.mouse_body_len), int(self.pos[1]+8)), 5)
        # if self.check_flag >= 6:
        #     self.check_flag == 6
        #     self.goal = False
        if self.goal is False and self.licking:
            pygame.draw.circle(self.map, 'red', (int(self.pos[0]+self.mouse_body_len), int(self.pos[1]+8)), 5)

    def update(self):
        # check speed
        self.speed -= 0.5
        if self.speed > 20:
            self.speed = 20
        if self.speed < speed_limit_low:
            self.speed = speed_limit_low

        # check average speed
        self.average_speed += self.speed
        self.average_speed /= 2

        # check position
        self.rotate_surface = rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

class MemoryTask:

    trial_types = ['T1', 'T2']
    
    def __init__(self, task_image):
        pygame.init()
        self.trial_type = self.trial_types[random.choice([0,1])]
        # self.trial_type = self.trial_types[0]
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('contextual moemory T1/2 task') 
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.animal_begin_location = begin_location['T1'].copy() if self.trial_type == 'T1' else begin_location['T2'].copy()
        self.animal = Mouse('mouse.png', task_image, self.animal_begin_location)
        self.animal.trial_type = self.trial_type
        self.frame_rate = 60
        self.mode = 0

    def update_by_action(self, action):
        if action == 0:
            self.animal.licking = False
        if action == 1:
            action = 3
        if action == 2:
            action = 3
        if action == 3:
            self.animal.licking = True
            self.animal.lick_times += 1
            self.animal.check_licking()

        self.animal.update()
        self.animal.check_trial_finish()
        
    def evaluate(self):
        reward = 0
        if self.animal.goal and self.animal.licking and self.animal.check_flag < 10:
            reward += 500
        if not self.animal.goal and self.animal.licking and self.animal.reward > 0:
            reward -= 300 * (1+F.relu(torch.tensor(self.animal.lick_times - 100)).item())
        if not self.animal.goal and self.animal.licking and self.animal.reward <= 0:
            reward -= 100 * (1+F.relu(torch.tensor(self.animal.lick_times - 100)).item())
        # reward -= self.animal.time_spent * 0.1
        self.animal.goal = False # reset goal status
        self.animal.reward += reward
        return reward

    def is_done(self):
        if self.animal.trial_finish:
            return True
        return False

    def observe(self):
        # Showing on GUI what the animal sees
        eye_sight = 45
        if self.animal.pos[0] > 358:
            self.animal.obs_arr = self.animal.maze_arr[57-15:57+30,   
                                                    int(self.animal.pos[0])+30:int(self.animal.pos[0])+30+eye_sight]
        elif self.animal.pos[0] <= 358:
            self.animal.obs_arr = self.animal.maze_arr[int(self.animal.pos[1])-15:int(self.animal.pos[1])+30,   
                                                    int(self.animal.pos[0])+30:int(self.animal.pos[0])+30+eye_sight]
        obs_arr = np.dot(self.animal.obs_arr[...,:3], rgb_weight) # convert to gray scale
        obs_arr = (obs_arr-obs_arr.mean())/obs_arr.std() # normalize
        # obs_arr += 0.2*np.random.randn(*obs_arr.shape)
        # self.animal.obs_arr = pygame.surfarray.make_surface(obs_arr)

        ### choice1: return observation of the animal ###
        return obs_arr.flatten()

        ### choice2：return the position of the animal ###
        # return np.array([self.animal.pos[0]]*eye_sight*eye_sight)

    def view(self):
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3

        self.screen.blit(self.animal.map, (0, 0))

        if self.mode == 1:
            self.screen.fill((0, 0, 0))

        self.animal.draw(self.screen, self.trial_type)

        pygame.display.flip()
        self.clock.tick(self.frame_rate)


def get_distance(p1, p2):
	return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))

def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image
