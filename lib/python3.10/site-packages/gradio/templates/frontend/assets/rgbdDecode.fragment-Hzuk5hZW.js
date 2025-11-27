import{j as r}from"./index-DtZsi8YU.js";import"./helperFunctions-DlLEDCEh.js";import"./index-BvFm9VBJ.js";import"./svelte/svelte.js";const e="rgbdDecodePixelShader",o=`varying vec2 vUV;uniform sampler2D textureSampler;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void) 
{gl_FragColor=vec4(fromRGBD(texture2D(textureSampler,vUV)),1.0);}`;r.ShadersStore[e]||(r.ShadersStore[e]=o);const m={name:e,shader:o};export{m as rgbdDecodePixelShader};
//# sourceMappingURL=rgbdDecode.fragment-Hzuk5hZW.js.map
