#version 450
#extension GL_EXT_buffer_reference : require

layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
	mat4 axisSwizzle;
	
	mat4 normal;
	vec3 cameraPos;
} ubo;

struct Particle
{
	vec4 position;
	vec4 rotation;
	vec4 deltaPos;
	vec4 deltaRotation;
};

layout(buffer_reference, std430) readonly buffer InParticleSSBO{ 
	Particle particles[];
};
layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 texCoord;
layout(location = 1) out vec3 fragPos;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 cameraPos;

layout( push_constant ) uniform constants
{	
	InParticleSSBO particles;
} pushConstants;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.axisSwizzle * (pushConstants.particles.particles[gl_InstanceIndex].position + vec4(inPosition, 1.0));
	fragPos = vec3(ubo.model * vec4(inPosition, 1.0));
	
	//normal = mat3(transpose(inverse(ubo.model))) * inNormal;
	normal = (ubo.normal * vec4(inNormal, 0.f)).xyz;
	
	texCoord = inTexCoord;
	cameraPos = ubo.cameraPos;
}
