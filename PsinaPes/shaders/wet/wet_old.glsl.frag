#version 450
#extension GL_EXT_buffer_reference : require

#include "../common/lib.glsl"

layout(location = 0) in vec2 TexCoords;
layout(location = 1) in vec3 WorldPos;
layout(location = 2) in mat3 TBN;

layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
	mat4 axisSwizzle;
	
	mat4 normal;
	vec3 cameraPos;
} ubo;


struct PointLight
{
    vec4 position;
	vec4 color;
};
layout(buffer_reference, std430) readonly buffer Lights{ 
	PointLight lights[];
};

layout(binding = 1) uniform sampler2D albedoMap;
layout(binding = 2) uniform sampler2D normalMap;
layout(binding = 3) uniform sampler2D metallicMap;
layout(binding = 4) uniform sampler2D roughnessMap;
layout(binding = 5) uniform sampler2D aoMap;

layout(binding = 6) uniform samplerCube irradianceMap;
layout(binding = 7) uniform samplerCube specularEnvironmentMap;
layout(binding = 8) uniform sampler2D brdfLUT;
 
layout(location = 0) out vec4 outColor;

layout( push_constant ) uniform constants
{	
	Lights lights;
	int lightCount;
} pushConstants;

void main()
{
	//vec3 albedo     = pow(texture(albedoMap, TexCoords).rgb, vec3(2.2));
	vec3 albedo = vec3(0.5);
    //float metallic  = texture(metallicMap, TexCoords).r;
    float metallic  = 0;
    //float roughness = texture(roughnessMap, TexCoords).r;
	float roughness = TexCoords.x;
    float ao        = texture(aoMap, TexCoords).r;
	
	float wetLevel = 1;
	//float wetLevel = texture(metallicMap, TexCoords).r;
	
	float waterEta = 1 / 1.33;
	vec3 waterF0 = vec3(0.04);
	float waterRoughness = 0.02;
	
	vec3 vertexNormal = TBN * vec3(0, 0, 1);
	
	vec3 N = TBN * (texture(normalMap, TexCoords).xyz * 2.0 - 1.0);
	//vec3 N = normalize(WorldPos); 
    vec3 V = normalize(ubo.cameraPos - WorldPos);
    vec3 R = reflect(-V, N); 
	
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0);
    for(int i = 0; i < pushConstants.lightCount; ++i) 
    {
		PointLight light = pushConstants.lights.lights[i];
		
        // calculate per-light radiance
        vec3 L = normalize(light.position.xyz - WorldPos);
        vec3 H = normalize(V + L);
        float distance = length(light.position.xyz - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = light.color.xyz * attenuation;

		vec3 LiLower = TransparentBTDF(L, radiance, vertexNormal, waterF0);
		
		vec3 wiLower = -normalize(refract(L, N, waterEta));
		vec3 woLower = -normalize(refract(V, N, waterEta));
		
		vec3 LoLower = MetallicWorkflowBRDF(wiLower, woLower, N, roughness, metallic, F0, albedo) * LiLower;

		vec3 LoDry = MetallicWorkflowBRDF(L, V, N, roughness, metallic, F0, albedo) * radiance;
				
		Lo += mix(LoDry, TransparentBTDF(-woLower, LoLower, -vertexNormal, waterF0), wetLevel);
    }   
    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness); 

	//vec3 kS = F; 
	//vec3 kD = 1.0 - kS;
	//vec3 irradiance = texture(irradianceMap, N).rgb;
	//vec3 diffuse    = irradiance * albedo;
	
	//const float mipLevels = textureQueryLevels(specularEnvironmentMap);
    //vec3 prefilteredColor = textureLod(specularEnvironmentMap, R, roughness * mipLevels).rgb;    
    //vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    //vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

	//vec3 ambient    = (kD * diffuse + specular) * ao;
	vec3 waterIndirect = BallsIndirectLighting(V, N, waterF0, waterF0, waterRoughness, ao, irradianceMap, specularEnvironmentMap, brdfLUT);
    vec3 color = Lo + waterIndirect * wetLevel;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2)); 

    outColor = vec4(color, 1.0);
	//outColor = vec4(vec2(max(dot(N, V), 0.0), roughness), 0, 1.0);
}
