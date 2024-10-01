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

layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;

struct PointLight
{
    vec4 position;
	vec4 color;
};
layout(buffer_reference, std430) readonly buffer Lights{ 
	PointLight lights[];
};

layout( push_constant ) uniform constants
{	
	Lights lights;
	int lightCount;
} pushConstants;


void main() 
{
    gl_Position = ubo.proj * ubo.view * ubo.axisSwizzle * (ubo.model * vec4(inPosition, 1.0) + vec4(pushConstants.lights.lights[gl_InstanceIndex].position.xyz, 0.0));		
}
