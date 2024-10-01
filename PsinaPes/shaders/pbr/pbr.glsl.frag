#version 450
#extension GL_EXT_buffer_reference : require

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


const float PI = 3.14159265359;
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
} 

void main()
{
	//vec3 albedo     = pow(texture(albedoMap, TexCoords).rgb, vec3(2.2));
	vec3 albedo     = vec3(1.0, 1.0, 0.0);
    //float metallic  = texture(metallicMap, TexCoords).r;
    float metallic  = 1;
    //float roughness = texture(roughnessMap, TexCoords).r;
    float roughness = 0;
    float ao        = texture(aoMap, TexCoords).r;

	vec3 N = TBN * (texture(normalMap, TexCoords).xyz * 2.0 - 1.0);
	//vec3 N = WorldPos; 

    vec3 V = normalize(ubo.cameraPos - WorldPos);
    vec3 R = reflect(-V, N); 
	
    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
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

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
           
        vec3 numerator    = NDF * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
        vec3 specular = numerator / denominator;
        
        // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals 
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metallic;	  

        // scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);        

        // add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }   
    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness); 

	vec3 kS = F; 
	vec3 kD = 1.0 - kS;
	vec3 irradiance = texture(irradianceMap, N).rgb;
	vec3 diffuse    = irradiance * albedo;
	
	const float mipLevels = textureQueryLevels(specularEnvironmentMap);
    vec3 prefilteredColor = textureLod(specularEnvironmentMap, R, roughness * mipLevels).rgb;    
    vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

	vec3 ambient    = (kD * diffuse + specular) * ao;
	    
    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2)); 

    outColor = vec4(color, 1.0);
	//outColor = vec4(vec2(max(dot(N, V), 0.0), roughness), 0, 1.0);
}
