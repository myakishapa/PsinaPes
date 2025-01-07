#version 450

layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
	mat4 axisSwizzle;
	
	mat4 normal;
	vec3 cameraPos;
} ubo;

layout(binding = 1) uniform sampler2D graphTexture;

layout(location = 0) out vec2 TexCoords;

layout( push_constant ) uniform constants
{	
	uint resolution;
	float textureScale;
} pushConstants;


void main() 
{	
	uint rowBias = gl_VertexIndex % 2;
	uint row = gl_VertexIndex / (pushConstants.resolution * 2);
	uint column = (gl_VertexIndex / 2) % pushConstants.resolution;
	
	uvec2 index2d = uvec2(column, row + rowBias);
	
	
	vec2 id2d = vec2(index2d) / pushConstants.resolution;
	
	vec3 position = vec3(id2d, 1);
	//vec3 position = vec3(id2d, texture(graphTexture, id2d).y * pushConstants.textureScale);
	
    gl_Position = ubo.proj * ubo.view * ubo.axisSwizzle * ubo.model * vec4(position, 1.0);
	
	TexCoords = id2d;
}
