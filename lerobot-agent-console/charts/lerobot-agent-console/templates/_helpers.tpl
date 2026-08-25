{{/* The resource name. Every object (Service, StatefulSet, selector labels, serviceName) uses
     this one value, so dev/test differ by a single values key and can never half-rename. */}}
{{- define "console.name" -}}
{{- default .Release.Name .Values.nameOverride -}}
{{- end -}}

{{/* Selector labels. StatefulSet.spec.selector is IMMUTABLE, and the live consoles were
     created with a bare `app: <name>` — so this must stay exactly that. Adding the usual
     helm.sh/chart or app.kubernetes.io/* labels here would make `helm upgrade` on the
     existing StatefulSets fail. Extra labels belong on the metadata, not the selector. */}}
{{- define "console.selectorLabels" -}}
app: {{ include "console.name" . }}
{{- end -}}

{{/* The console's HTTP port. Defined once: the Service publishes it and the APIG Ingress routes
     to it, and those two silently disagreeing is a 503 with nothing in the logs. */}}
{{- define "console.port" -}}
{{- default 8080 .Values.port -}}
{{- end -}}

{{/* ---------------------------------------------------------------------------------------
     APIG. Field names and semantics are kept identical to the RLinf chart on purpose, so the
     two products present one story: same keys, same modes, same failure messages. Port them
     together when either changes.
     --------------------------------------------------------------------------------------- */}}

{{/* The APIGInstance OBJECT's name — not the gateway's. When adopting, the platform already
     has one named after the gateway id, and that is what the binding annotation must point at;
     a name of our choosing would refer to nothing. */}}
{{- define "console.apig.instanceObjectName" -}}
{{- if .Values.apig.create }}
{{- printf "%s-apig" (include "console.name" .) }}
{{- else }}
{{- printf "%s-apig-instance" .Values.apig.existingId }}
{{- end }}
{{- end -}}

{{- define "console.apig.ingressName" -}}
{{- default (printf "%s-apig" (include "console.name" .)) .Values.apig.ingressName -}}
{{- end -}}

{{- define "console.apig.ingressClassName" -}}
{{- if .Values.apig.ingressClassName }}
{{- .Values.apig.ingressClassName }}
{{- else if .Values.apig.create }}
{{- printf "%s-apig" (include "console.name" .) }}
{{- else }}
{{- fail "apig.ingressClassName is required when apig.create=false: it must match the ingress class the adopted gateway declares, and there is no default that could be correct." }}
{{- end }}
{{- end -}}

{{- define "console.apig.host" -}}
{{- default (printf "%s.apig.local" (include "console.name" .)) .Values.apig.host -}}
{{- end -}}

{{/* Binding annotations, derived from existingId — so they appear in ADOPT mode only.

     In create mode there is deliberately nothing here. The gateway does not have an id until it
     has been provisioned, and the APIGInstance this chart renders claims the Ingress from the
     other side, by watching this namespace and this ingress class. That is what makes install a
     single step: an earlier version of this chart asked the operator to read status.id back into
     values and upgrade again. */}}
{{- define "console.apig.annotations" -}}
{{- with .Values.apig.existingId }}
ingress.vke.volcengine.com/apig-instance-name: {{ include "console.apig.instanceObjectName" $ | quote }}
ingress.vke.volcengine.com/loadbalancer-id: {{ . | quote }}
{{- end }}
{{- with .Values.apig.annotations }}
{{/* ⚠️ Helm DEEP-MERGES maps across values files: `annotations: {}` in an override does NOT
     clear the parent's entries. That silently gave the test console dev's gateway, and its
     public IP stopped answering entirely (000, not even a 5xx). */}}
{{- toYaml . }}
{{- end }}
{{- end -}}

{{- define "console.apig.validate" -}}
{{- if .Values.apig.enabled }}
{{- if .Values.apig.create }}
{{- if .Values.apig.existingId }}
{{- fail "apig.existingId must be empty when apig.create=true. The provisioned gateway's id is reported in the APIGInstance's status.id and the Ingress binds by ingress class, so nothing needs it back. Setting it writes spec.id, which is immutable — the admission webhook then rejects every upgrade with 'spec.id: Forbidden: forbidden to update'. Use existingId only with apig.create=false." }}
{{- end }}
{{- if not .Values.apig.subnetIds }}
{{- fail "apig.create is true but apig.subnetIds is empty — a new gateway needs subnets in this cluster's VPC. Either list them, or set apig.create=false and point apig.existingId at a gateway that already exists." }}
{{- end }}
{{- else }}
{{- if not .Values.apig.existingId }}
{{- fail "apig.existingId is required when apig.create=false: set it to the gateway's instance id from the APIG console, or set apig.create=true to provision a new gateway." }}
{{- end }}
{{- end }}
{{- end }}
{{- end -}}
