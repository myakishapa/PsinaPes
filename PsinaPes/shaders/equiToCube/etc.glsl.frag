#version 450

layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 proj;
	mat4 axisSwizzle;
	mat4 view[6];
} ubo;

layout(binding = 1) uniform sampler2D equirectangularMap;

layout(location = 0) in vec3 WorldPos;

layout(location = 0) out vec4 outColor;

const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

void main()
{
	vec2 uv = SampleSphericalMap(normalize(WorldPos));
    vec3 color = texture(equirectangularMap, uv).rgb;
    
    outColor = vec4(color, 1.0);
}
