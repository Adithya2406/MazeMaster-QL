from pyglet.gl import *

def setup_basic_lighting(program):
    glUseProgram(program)
    glUniform3f(glGetUniformLocation(program, "lightDir"),
                -0.4, -1.0, -0.4)
    glUniform3f(glGetUniformLocation(program, "lightColor"),
                1.0, 1.0, 1.0)
