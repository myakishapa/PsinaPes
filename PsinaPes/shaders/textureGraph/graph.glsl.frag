#version 450
#extension GL_EXT_buffer_reference : require

#include "../common/lib.glsl"

layout(location = 0) in vec2 TexCoords;

layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
	mat4 axisSwizzle;
	
	mat4 normal;
	vec3 cameraPos;
} ubo;

layout(binding = 1) uniform sampler2D graphTexture;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 debugColor1;
layout(location = 2) out vec4 debugColor2;
layout(location = 3) out vec4 debugColor3;
layout(location = 4) out vec4 debugColor4;
layout(location = 5) out vec4 debugColor5;
layout(location = 6) out vec4 debugColor6;
layout(location = 7) out vec4 debugColor7;

void main()
{
	
    outColor = texture(graphTexture, TexCoords);
}
