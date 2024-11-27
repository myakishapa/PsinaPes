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
float GeometrySchlickGGX(float NdotW, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotW;
    float denom = NdotW * (1.0 - k) + k;

    return nom / denom;
}
float GeometrySmith(vec3 N, vec3 Wo, vec3 Wi, float roughness)
{
    float NdotWo = max(dot(N, Wo), 0.0);
    float NdotWi = max(dot(N, Wi), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotWo, roughness);
    float ggx1 = GeometrySchlickGGX(NdotWi, roughness);

    return ggx1 * ggx2;
}
vec3 fresnelSchlick(float HdotWo, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - HdotWo, 0.0, 1.0), 5.0);
}
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
} 
vec3 SpecularCookTorrance(vec3 Wi, vec3 Wo, vec3 N, float roughness, vec3 F0)
{
	vec3 H = normalize(Wi + Wo);

	float NDF = DistributionGGX(N, H, roughness);   
    float G   = GeometrySmith(N, Wo, Wi, roughness);      
    vec3 F    = fresnelSchlick(max(dot(H, Wo), 0.0), F0);
           
    vec3 numerator    = NDF * G * F; 
    float denominator = 4.0 * max(dot(N, Wo), 0.0) * max(dot(N, Wi), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;
	
	float NdotL = max(dot(N, Wi), 0.0);        
	
	return specular * NdotL;
}

vec3 MetallicWorkflowBRDF(vec3 wi, vec3 wo, vec3 N, float roughness, float metallic, vec3 F0, vec3 albedo)
{
	vec3 H = normalize(wi + wo);

	float NDF = DistributionGGX(N, H, roughness);   
    float G   = GeometrySmith(N, wo, wi, roughness);      
    vec3 F    = fresnelSchlick(max(dot(H, wo), 0.0), F0);
           
    vec3 numerator    = NDF * G * F; 
    float denominator = 4.0 * max(dot(N, wo), 0.0) * max(dot(N, wi), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;
	
	//return wi;
	//return vec3(G, NDF, dot(N, wi));

    vec3 kS = F;
        
    vec3 kD = vec3(1.0) - kS;

    kD *= 1.0 - metallic;	
	
	float NdotL = max(dot(N, wi), 0.0);        
	
	vec3 brdf = (kD * albedo / PI + specular)  * NdotL;
	return brdf;
}

vec3 IndirectLighting(vec3 Wo, vec3 N, vec3 F0, vec3 albedo, float roughness, float ao, samplerCube irradianceMap, samplerCube specularEnvironmentMap, sampler2D brdfLUT)
{
	vec3 F = fresnelSchlickRoughness(max(dot(N, Wo), 0.0), F0, roughness); 

	vec3 kS = F; 
	vec3 kD = 1.0 - kS;
	vec3 irradiance = texture(irradianceMap, N).rgb;
	vec3 diffuse    = irradiance * albedo;
	
	vec3 R = reflect(-Wo, N);
	
	const float mipLevels = textureQueryLevels(specularEnvironmentMap);
    vec3 prefilteredColor = textureLod(specularEnvironmentMap, R, roughness * mipLevels).rgb;    
    vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, Wo), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

	vec3 ambient  = (kD * diffuse + specular) * ao;
	return ambient;
}

float BallsDistributionGGX(float NdotH, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    NdotH = max(NdotH, 0.0);
	float NdotH2 = NdotH*NdotH;
	
    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}    

vec3 BallsIndirectLighting(vec3 Wo, vec3 N, vec3 F0, vec3 albedo, float roughness, float ao, samplerCube irradianceMap, samplerCube specularEnvironmentMap, sampler2D brdfLUT)
{
	vec3 F = fresnelSchlick(max(dot(N, Wo), 0.0), F0); 

	vec3 kS = F; 
	vec3 kD = 1.0 - kS;
	vec3 irradiance = texture(irradianceMap, N).rgb;
	vec3 diffuse    = irradiance * albedo;
	
	vec3 R = reflect(-Wo, N);
	
	const float mipLevels = textureQueryLevels(specularEnvironmentMap);
    vec3 I = textureLod(specularEnvironmentMap, R, roughness * mipLevels).rgb;    
    vec3 numerator = F * GeometrySmith(N, Wo, R, roughness);
    float denominator = 4.0 * max(dot(N, Wo), 0.0) * max(dot(N, R), 0.0) + 0.0001;

	vec3 specular = I * numerator / denominator;

	vec3 ambient  = (kD * diffuse + specular) * ao;
	return ambient;
}

vec3 TransparentBTDF(vec3 wi, vec3 Li, vec3 N, vec3 F0)
{	
	vec3 R = fresnelSchlick(max(dot(wi, N), 0.0), F0);
	vec3 T = 1.0 - R;
	
	float NdotWi = max(dot(N, wi), 0.0);        

	return Li * T * NdotWi;
}

float FresnelSchlickGeneric(float HdotWo)
{
	return pow(clamp(1.0 - HdotWo, 0.0, 1.0), 5.0);
}

float LobeAngleToRadiantIntensityMipLevel(float theta, float levels)
{
	return (theta - 1e-4) / PI * 2 * (levels - 1);
}

vec3 MyakishIndirectLighting(vec3 Wo, vec3 N, vec3 F0, float roughness, float ao, sampler2D lobeSolidAngle, sampler2D averagedBRDF, samplerCube radiantIntensity, out vec4 debug2, out vec4 debug3, out vec4 debug4, out vec4 debug5, out vec4 debug6, out vec4 debug7)
{
	float NdotWo = dot(N, Wo);
	
	float lobeAngle = texture(lobeSolidAngle, vec2(roughness, NdotWo)).x;
	
	vec3 R = reflect(-Wo, N);
	float mipLevel = LobeAngleToRadiantIntensityMipLevel(lobeAngle, textureQueryLevels(radiantIntensity));
	
	vec3 I = textureLod(radiantIntensity, R, mipLevel).rgb;    
	
	vec4 BRDF = texture(averagedBRDF, vec2(roughness, NdotWo));
	
	float D = BRDF.x;
	float G = BRDF.y;
	float GenericF = BRDF.z;
	float NdotWi = BRDF.w;
	//float NdotWi = dot(N, R);
	
	vec3 F = GenericF * (1.0 - F0) + F0;
	
	vec3 numerator = D * G * F;
	float denominator = 4.0 * NdotWo * NdotWi + 1e-4;
	
	vec3 specular = I * numerator / denominator;
	
	vec3 ambient = specular * ao;
	
	debug3 = vec4(I, 1.0);
	debug4 = vec4(numerator, denominator);
	
	debug5 = vec4(roughness, NdotWo, 0, 0);
	debug6 = vec4(numerator / denominator, 0);
	debug7 = BRDF;
	
	return ambient;
	//return numerator / denominator;
	//return vec3(mipLevel / float(textureQueryLevels(radiantIntensity)), lobeAngle, 0);
}


vec3 MyakishIndirectLighting2(vec3 Wo, vec3 N, vec3 F0, float roughness, float ao, sampler2D lobeAngles, sampler2D integratedBRDF, samplerCube integratedRadiance, out vec4 debug2, out vec4 debug3, out vec4 debug4, out vec4 debug5, out vec4 debug6, out vec4 debug7)
{
	float NdotWo = dot(N, Wo);
	
	vec2 angles = texture(lobeAngles, vec2(roughness, NdotWo)).xy;
	float elevation = angles.x;
	float angle = angles.y;
	
	vec3 NcrossWo = normalize(cross(Wo, N));
	
	vec3 base = cross(NcrossWo, N);
	
	vec3 radianceSample = cos(elevation) * base + sin(elevation) * N;
	
	float mipLevel = LobeAngleToRadiantIntensityMipLevel(angle, textureQueryLevels(integratedRadiance));
	
	vec3 radiance = textureLod(integratedRadiance, radianceSample, mipLevel).rgb;    
	
	float preintegratedBRDF = texture(integratedBRDF, vec2(roughness, NdotWo)).x;
	
		
	vec3 BRDF = F0 + preintegratedBRDF * (1.0 - F0);
	
	vec3 specular = radiance * BRDF;
	
	vec3 ambient = specular * ao;
	
	debug2 = vec4(BRDF, 0);
	debug3 = vec4(elevation, angle, 0, 0);
	debug4 = vec4(radianceSample, 0);
	debug5 = vec4(radiance, 0);
	debug6 = vec4(specular, 0);
	debug7 = vec4(preintegratedBRDF, 0, 0, 0);
	
	return ambient;
	//return numerator / denominator;
	//return vec3(mipLevel / float(textureQueryLevels(radiantIntensity)), lobeAngle, 0);
}
