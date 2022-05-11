from pygame.locals import *
import pygame
import numpy as np

class Player:
    x = 58
    y = 57 #119.2
    # y = 119.2
    speed = 0.05
 
    def moveRight(self):
        self.x = self.x + self.speed
 
    def moveLeft(self):
        self.x = self.x - self.speed
 
    def moveUp(self):
        self.y = self.y - self.speed
 
    def moveDown(self):
        self.y = self.y + self.speed
 
class Maze:
    def __init__(self):
       self.M = 10
       self.N = 8
       self.maze = [ 1,1,1,1,1,1,1,1,1,1,
                     1,0,0,0,0,0,0,0,0,1,
                     1,0,0,0,0,0,0,0,0,1,
                     1,0,1,1,1,1,1,1,0,1,
                     1,0,1,0,0,0,0,0,0,1,
                     1,0,1,0,1,1,1,1,0,1,
                     1,0,0,0,0,0,0,0,0,1,
                     1,1,1,1,1,1,1,1,1,1,]

    def draw(self,display_surf,image_surf):
       bx = 0
       by = 0
       for i in range(0,self.M*self.N):
           if self.maze[ bx + (by*self.M) ] == 1:
               display_surf.blit(image_surf,( bx * 44 , by * 44))
      
           bx = bx + 1
           if bx > self.M-1:
               bx = 0 
               by = by + 1


class App:
 
    windowWidth = 800
    windowHeight = 250
 
    def __init__(self):
        self._running = True
        self.surface = None
        self.image_mouse = None
        self._block_surf = None
        self.player = Player()
        self.maze = Maze()
 
    def on_init(self):
        pygame.init() 
        self.surface = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE) 
        pygame.display.set_caption('contextual moemory task') 
        self._running = True
        self.image_mouse = pygame.transform.scale(pygame.image.load("mouse.png").convert(), (30,16)) 
        # self._block_surf = pygame.transform.scale(pygame.image.load("task.png").convert(), (100,20))
        maze_image = pygame.transform.scale(pygame.image.load("task.png"), (self.windowWidth,self.windowHeight)) 
        self.maze_image = maze_image.convert() 
        self.maze_arr = pygame.surfarray.array3d(maze_image).swapaxes(0,1) 
        mouse_body_len = 28
        pygame.draw.line(self.maze_image, 'black', (412+mouse_body_len,0), (412+mouse_body_len,250), 2)
        pygame.draw.line(self.maze_image, 'black', (471+mouse_body_len,0), (471+mouse_body_len,250), 2)
        pygame.draw.line(self.maze_image, 'black', (557+mouse_body_len,0), (557+mouse_body_len,250), 2)
        pygame.draw.line(self.maze_image, 'black', (616+mouse_body_len,0), (616+mouse_body_len,250), 2)
        self.font = pygame.font.SysFont("Arial", 15) 

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False
 
    def on_loop(self):
        pass
    
    def on_render(self):
        ## draw background, maze and mouse (must draw in this order)
        self.surface.fill('white')
        self.surface.blit(self.maze_image,(0,0))
        self.surface.blit(self.image_mouse,(self.player.x,self.player.y))
        # self.maze.draw(self.surface, self._block_surf)  # draw programmed maze      

        ## show animal position
        text = self.font.render("pos:{:.2f},{:.2f}".format(self.player.x, self.player.y), True, 'black')
        text_rect = text.get_rect()
        text_rect.center = (50, 10)
        self.surface.blit(text, text_rect)

        ## show observation of the mouse
        obs_arr = self.maze_arr[int(self.player.y)-15:int(self.player.y)+30, int(self.player.x)+30:int(self.player.x)+130]
        obs_surf = pygame.surfarray.make_surface(obs_arr.swapaxes(0,1))
        self.surface.blit(obs_surf, (100, 200))

        pygame.display.flip()
 
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            if (keys[K_RIGHT]):
                self.player.moveRight()
 
            if (keys[K_LEFT]):
                self.player.moveLeft()
 
            if (keys[K_UP]):
                self.player.moveUp()
 
            if (keys[K_DOWN]):
                self.player.moveDown()
 
            if (keys[K_ESCAPE]):
                self._running = False
 
            self.on_loop()
            self.on_render()
        self.on_cleanup()

if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()