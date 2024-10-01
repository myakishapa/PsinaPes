#version 450

layout(location = 0) in vec2 texCoord;
layout(location = 1) in vec3 fragPos;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 cameraPos;

struct PointLight
{
    vec3 position;
	vec3 color;
};

layout(binding = 1) uniform Light {
	PointLight point[2];
} lightUbo;

layout(binding = 2) uniform sampler2D textureHui;

vec3 PointLighting(PointLight light)
{
	vec3 norm = normalize(normal);
	vec3 lightDir = normalize(light.position - fragPos);
	vec3 viewDir = normalize(cameraPos - fragPos);
	vec3 reflectDir = reflect(-lightDir, norm); 
	
	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuse = diff * light.color;
	
	float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * light.color;
	
	float specularStrength = 0.5;
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
	vec3 specular = specularStrength * spec * light.color;
	
	return ambient + diffuse + specular;
}

layout(location = 0) out vec4 outColor;

void main()
{
	vec3 total = vec3(0);

	for(int i = 0; i < 2; i++)
		total += PointLighting(lightUbo.point[i]);

	vec4 objectColor = texture(textureHui, texCoord);
		
	outColor = objectColor * vec4(total, 1.f);
	//outColor = vec4(normal, 1.0);
	//outColor = objectColor;
	//outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
