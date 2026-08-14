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
