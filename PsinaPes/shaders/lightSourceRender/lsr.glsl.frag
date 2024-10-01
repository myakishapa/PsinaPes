#version 450

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(1);
	//outColor = vec4(vec2(max(dot(N, V), 0.0), roughness), 0, 1.0);
}
