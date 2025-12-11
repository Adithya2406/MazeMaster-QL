# core/shaders.py

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragNormal;
out vec2 fragUV;
out vec3 fragWorld;

void main() {
    vec4 worldPos = model * vec4(inPos, 1.0);
    fragWorld = worldPos.xyz;
    fragNormal = mat3(model) * inNormal;
    fragUV = inUV;
    gl_Position = projection * view * worldPos;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 fragNormal;
in vec3 fragWorld;
in vec2 fragUV;

uniform sampler2D tex;
uniform int isGoal;

out vec4 FragColor;

void main() {
    vec3 lightDir = normalize(vec3(0.4, 1.0, 0.3));
    float diff = max(dot(normalize(fragNormal), lightDir), 0.1);

    vec3 base = texture(tex, fragUV).rgb * diff;

    if (isGoal == 1) {
        base = vec3(0.2, 0.4, 1.0) + diff * 0.5;   // blue glow
    }

    FragColor = vec4(base, 1.0);
}
"""
