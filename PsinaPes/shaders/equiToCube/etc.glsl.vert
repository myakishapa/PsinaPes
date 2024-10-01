#version 450

layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 proj;
	mat4 axisSwizzle;
	mat4 view[6];
} ubo;

layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;

layout(location = 0) out vec3 WorldPos;

void main() 
{
	WorldPos = inPosition;
}
