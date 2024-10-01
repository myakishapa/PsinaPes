#version 450

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

layout(location = 0) out vec2 TexCoords;
layout(location = 1) out vec3 WorldPos;
layout(location = 2) out mat3 TBN;

void main() 
{
    gl_Position = ubo.proj * ubo.view * ubo.axisSwizzle * ubo.model * vec4(inPosition, 1.0);
	
	WorldPos = vec3(ubo.model * vec4(inPosition, 1.0));
	TexCoords = inTexCoord;
	
	mat3 normalMatrix = mat3(ubo.normal);
	vec3 T = normalize(normalMatrix * inTangent);
    vec3 N = normalize(normalMatrix * inNormal);
    //T = normalize(T - dot(T, N) * N);
    vec3 B = -cross(N, T);
	
	//TBN = transpose(mat3(T, B, N));    
	TBN = mat3(T, B, N);    
}
