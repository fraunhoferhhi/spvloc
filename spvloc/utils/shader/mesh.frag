#version 330 core

in vec3 frag_position;
in vec3 frag_normal;

#ifdef COLOR_0_LOC
in vec4 color_multiplier;
#endif

out vec4 frag_color;

void main()
{
    vec3 normal = normalize(frag_normal);

    frag_color = vec4(normal * 0.5 + 0.5, 1.0);
#ifdef COLOR_0_LOC
    frag_color.a = color_multiplier.x;
#endif
}