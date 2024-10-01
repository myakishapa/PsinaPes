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

layout(location = 0) out vec2 texCoord;
layout(location = 1) out vec3 fragPos;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 cameraPos;

const float pi = 3.1415;
const float tau = pi * 2;
vec2 DirToEquirectangular(vec4 dir)
{
	dir = normalize(dir);
	return vec2(atan(dir.y, dir.x) / tau + 0.5, asin(dir.z) / pi + 0.5);
}

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.axisSwizzle * ubo.model * vec4(inPosition, 1.0);
	vec4 dir = ubo.view * ubo.axisSwizzle * ubo.model * vec4(inPosition, 1.0);
	vec2 equir = DirToEquirectangular(dir);
	//gl_Position = vec4((equir * 2) - vec2(1), length(dir) / 100.f, 1.f);


	fragPos = vec3(ubo.model * vec4(inPosition, 1.0));
	
	//normal = mat3(transpose(inverse(ubo.model))) * inNormal;
	normal = (ubo.normal * vec4(inNormal, 0.f)).xyz;
	
	texCoord = inTexCoord;
	cameraPos = ubo.cameraPos;
}
