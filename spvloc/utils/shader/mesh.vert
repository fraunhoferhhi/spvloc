#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = NORMAL_LOC) in vec3 normal;
layout(location = INST_M_LOC) in mat4 inst_m;
#ifdef COLOR_0_LOC
layout(location = COLOR_0_LOC) in vec4 color_0;
#endif

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Outputs
out vec3 frag_position;
out vec3 frag_normal;
#ifdef COLOR_0_LOC
out vec4 color_multiplier;
#endif

void main()
{
    gl_Position = P * V * M * inst_m * vec4(position, 1);
    frag_position = vec3(M * inst_m * vec4(position, 1.0));

    mat3 N  = transpose(mat3(V));
    N[2] = -N[2]; 
    frag_normal = normalize(normal) * N;

#ifdef COLOR_0_LOC
    color_multiplier = color_0;
#endif
}