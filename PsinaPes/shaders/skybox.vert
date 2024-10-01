#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragPos;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
	mat4 axisSwizzle;
} ubo;

void main() {
	gl_Position = ubo.proj * ubo.view * ubo.axisSwizzle * vec4(inPosition, 1.0);
	fragPos = vec3(ubo.axisSwizzle * vec4(inPosition, 1.0));
}