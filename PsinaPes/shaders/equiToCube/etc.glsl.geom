#version 450

layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 proj;
	mat4 axisSwizzle;
	mat4 view[6];
} ubo;

layout (triangles) in;
layout (triangle_strip, max_vertices = 18) out;

layout(location = 0) in vec3 WorldPos[];

layout(location = 0) out vec3 outWorldPos;

void main() 
{
	for(int i = 0; i < 6; i++)
	{
		gl_Layer = i;
		for(int j = 0; j < 3; j++)
		{
			gl_Position = ubo.proj * ubo.view[i] * ubo.axisSwizzle * vec4(WorldPos[j], 1.0);
			outWorldPos = WorldPos[j];
			EmitVertex();
		}
		EndPrimitive();
	}
}
