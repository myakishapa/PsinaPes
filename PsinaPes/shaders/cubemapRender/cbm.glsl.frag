#version 450

layout(location = 0) in vec3 fragPos;

layout(binding = 1) uniform samplerCube textureHui;

layout(location = 0) out vec4 outColor;

void main()
{
	outColor = texture(textureHui, fragPos);
	//outColor = vec4(normalize(fragPos), 1);
}
